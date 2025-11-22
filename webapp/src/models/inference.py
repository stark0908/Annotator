# src/models/inference.py
import os
import json
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import mobilenet_v3_large
from PIL import Image
import numpy as np

# -------------------------
# minimal image <-> tensor
# -------------------------
def pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    img = np.array(img_pil)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = img[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img

# -------------------------
# Detector dataset (for training)
# -------------------------
class CocoLikeDetectionDataset(Dataset):
    def __init__(self, coco_json_path: str, img_root: str):
        self.img_root = img_root
        with open(coco_json_path, 'r') as f:
            self.coco = json.load(f)
        self.imgs = {i['id']: i for i in self.coco['images']}
        self.anns_per_img = defaultdict(list)
        for ann in self.coco['annotations']:
            self.anns_per_img[ann['image_id']].append(ann)
        self.cat_ids = sorted([c['id'] for c in self.coco['categories']])
        self.cat_id_to_contiguous = {cid: idx+1 for idx, cid in enumerate(self.cat_ids)}
        self.image_ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.imgs[img_id]
        path = os.path.join(self.img_root, info['file_name'])
        img = Image.open(path).convert('RGB')
        img_tensor = pil_to_tensor(img)
        annots = self.anns_per_img.get(img_id, [])
        boxes, labels, iscrowd, areas = [], [], [], []
        for a in annots:
            x, y, w, h = a['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            iscrowd.append(a.get('iscrowd', 0))
            areas.append(a.get('area', w*h))
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------------
# Detector factory
# -------------------------
def get_fasterrcnn(num_classes: int, pretrained_backbone: bool = True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# training wrapper
def train_detector(coco_json: str, img_root: str, out_path: str, device: str = None,
                   epochs: int = 100, batch_size: int = 2, lr: float = 1e-4):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using",device)
    dataset = CocoLikeDetectionDataset(coco_json, img_root)
    num_classes = 2
    model = get_fasterrcnn(num_classes).to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, targets in data_loader:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        print(f"[Detector] Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'num_classes': num_classes}, out_path)
    return out_path

def load_detector(checkpoint_path: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training detector on",device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get('num_classes')
    model = get_fasterrcnn(num_classes, pretrained_backbone=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

def inference_detector(model, image_path: str, device: str = None, score_thresh: float = 0.3):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using",device)
    img = Image.open(image_path).convert('RGB')
    img_tensor = pil_to_tensor(img).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])
    out = outputs[0]
    boxes = out['boxes'].cpu()
    scores = out['scores'].cpu()
    labels = out['labels'].cpu()
    results = []
    for b, s, l in zip(boxes, scores, labels):
        if s.item() < score_thresh:
            continue
        x1, y1, x2, y2 = b.tolist()
        results.append({'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s.item()), 'label_idx': int(l.item())})
    return results

# -------------------------
# Prototypical classifier (embedding net + training)
# -------------------------
class RoiAnnotationDataset(Dataset):
    def __init__(self, coco_json: str, img_root: str, category_id_to_idx: Dict[int, int]=None, transform=None):
        with open(coco_json, 'r') as f:
            self.coco = json.load(f)
        self.imgs = {i['id']: i for i in self.coco['images']}
        self.anns = self.coco['annotations']
        self.categories = {c['id']: c['name'] for c in self.coco['categories']}
        if category_id_to_idx is None:
            unique_cats = sorted([c['id'] for c in self.coco['categories']])
            self.cat_to_idx = {cid: i for i, cid in enumerate(unique_cats)}
        else:
            self.cat_to_idx = category_id_to_idx
        self.img_root = img_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
        ])
        self.indices_per_class = defaultdict(list)
        for i, ann in enumerate(self.anns):
            cid = ann['category_id']
            if cid in self.cat_to_idx:
                self.indices_per_class[self.cat_to_idx[cid]].append(i)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_info = self.imgs[ann['image_id']]
        path = os.path.join(self.img_root, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        x, y, w, h = [int(v) for v in ann['bbox']]
        crop = img.crop((x, y, x+w, y+h))
        crop = self.transform(crop)
        tensor = pil_to_tensor(crop)
        label = self.cat_to_idx[ann['category_id']]
        return tensor, label

    def get_indices_for_class(self, class_idx: int):
        return self.indices_per_class[class_idx]

class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim: int = 256, pretrained: bool = True):
        super().__init__()
        base = mobilenet_v3_large(weights="IMAGENET1K_V2")
        modules = list(base.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        feat_dim = 960
        self.fc = nn.Linear(feat_dim, emb_dim)
        self.head = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, emb_dim))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class PrototypicalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prototypes, query_embeddings, query_labels):
        dists = torch.cdist(query_embeddings, prototypes)
        logits = -dists
        loss = F.cross_entropy(logits, query_labels)
        preds = logits.argmax(dim=1)
        acc = (preds == query_labels).float().mean()
        return loss, acc

class EpisodeSampler:
    def __init__(self, dataset: RoiAnnotationDataset, n_way: int = 5, k_shot: int = 5, q_query: int = 5):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.classes = list(dataset.indices_per_class.keys())

    def sample_episode(self):
        chosen = random.sample(self.classes, self.n_way)
        support_x, support_y = [], []
        query_x, query_y = [], []
        for i, cls in enumerate(chosen):
            indices = self.dataset.get_indices_for_class(cls)
            if len(indices) < self.k_shot + self.q_query:
                indices = indices * math.ceil((self.k_shot + self.q_query) / max(1,len(indices)))
            sampled = random.sample(indices, self.k_shot + self.q_query)
            support_inds = sampled[:self.k_shot]
            query_inds = sampled[self.k_shot:]
            for si in support_inds:
                x, _ = self.dataset[si]
                support_x.append(x)
                support_y.append(i)
            for qi in query_inds:
                x, _ = self.dataset[qi]
                query_x.append(x)
                query_y.append(i)
        support_x = torch.stack(support_x, dim=0)
        query_x = torch.stack(query_x, dim=0)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_y = torch.tensor(query_y, dtype=torch.long)
        return support_x, support_y, query_x, query_y

def train_prototypical(coco_json: str, img_root: str, out_path: str,
                       device: str = None,
                       n_way: int = 5, k_shot: int = 5, q_query: int = 5,
                       episodes_per_epoch: int = 100, epochs: int = 100, lr: float = 1e-4):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training classifier on",device,epochs)
    dataset = RoiAnnotationDataset(coco_json, img_root)
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, q_query=q_query)
    model = EmbeddingNet(emb_dim=256, pretrained=True).to(device)
    criterion = PrototypicalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for ep in range(episodes_per_epoch):
            support_x, support_y, query_x, query_y = sampler.sample_episode()
            support_x = support_x.to(device); query_x = query_x.to(device)
            support_y = support_y.to(device); query_y = query_y.to(device)
            emb_support = model(support_x)
            emb_query = model(query_x)
            prototypes = []
            for i in range(n_way):
                mask = (support_y == i)
                proto = emb_support[mask].mean(dim=0).to(device)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes, dim=0).to(device)
            loss, acc = criterion(prototypes, emb_query, query_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); epoch_acc += acc.item()
        print(f"[Proto] Epoch {epoch+1}/{epochs} loss={epoch_loss/episodes_per_epoch:.4f} acc={epoch_acc/episodes_per_epoch:.4f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'emb_dim': 256}, out_path)
    return out_path

def load_prototypes(proto_ckpt: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using",device)
    state = torch.load(proto_ckpt, map_location=device)
    model = EmbeddingNet(emb_dim=state.get('emb_dim', 256), pretrained=False)
    model.load_state_dict(state['model_state_dict'])
    model.to(device).eval()
    return model

# -------------------------
# Build prototypes from existing annotations (mean embedding per category)
# -------------------------
def build_prototypes_from_annotations(embedding_model: EmbeddingNet, coco_json: str, img_root: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using",device)
    dataset = RoiAnnotationDataset(coco_json, img_root)
    embedding_model = embedding_model.to(device).eval()
    emb_per_class = defaultdict(list)
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, label = dataset[idx]
            x = x.unsqueeze(0).to(device)
            emb = embedding_model(x)
            emb_per_class[label].append(emb.squeeze(0).cpu())
    prototypes = {}
    for cls, embs in emb_per_class.items():
        prototypes[cls] = torch.stack(embs, dim=0).mean(dim=0)
    return prototypes, dataset.cat_to_idx

# -------------------------
# Predict: run detector -> crop -> compute embedding -> match prototypes
# -------------------------
def run_detector_inference_and_classify(
    image_path: str, 
    detector_ckpt: str, 
    proto_ckpt: str, 
    coco_json: str, 
    img_root: str, 
    device: str = None, 
    score_thresh: float = 0.3
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load models
    detector = load_detector(detector_ckpt, device=device)
    emb_model = load_prototypes(proto_ckpt, device=device)

    # Build prototypes
    prototypes, cat_to_idx = build_prototypes_from_annotations(
        emb_model, coco_json, img_root, device=device
    )

    # Reverse map to COCO category ids
    idx_to_catid = {v: k for k, v in cat_to_idx.items()}
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    catid_to_name = {c['id']: c['name'] for c in coco['categories']}

    # Run detector
    dets = inference_detector(detector, image_path, device=device, score_thresh=score_thresh)

    # Load image
    pil_img = Image.open(image_path).convert('RGB')
    W, H = pil_img.size

    results = []
    for det in dets:
        x, y, w, h = det['bbox']

        # ---- ✅ Sanitize and clamp bbox ----
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))

        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid box: {x,y,w,h}")
            continue

        # ---- ✅ Crop and classify ----
        crop = pil_img.crop((x1, y1, x2, y2))
        crop = transforms.Resize((224,224))(crop)
        x_tensor = pil_to_tensor(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = emb_model(x_tensor).squeeze(0)

        best_cls = None
        best_sim = -1e9
        for cls, proto in prototypes.items():
            sim = F.cosine_similarity(emb.unsqueeze(0), proto.unsqueeze(0).to(device)).item()
            if sim > best_sim:
                best_sim = sim
                best_cls = cls

        coco_catid = idx_to_catid[best_cls]
        results.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],   # ✅ Ensure clean
            'score': det['score'],
            'category_id': int(coco_catid),
            'category_name': catid_to_name[int(coco_catid)],
            'pred_score': float(best_sim)
        })

    return results


# -------------------------
# Simple utility: render PIL with boxes (used by front-end viewer if needed)
# -------------------------
from PIL import ImageDraw, ImageFont

def render_detections_on_image(image_path: str, detections: List[Dict], out_path: str = None):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for det in detections:
        x, y, w, h = det['bbox']
        label = det.get('category_name', str(det.get('category_id', '')))
        score = det.get('pred_score', det.get('score', 0.0))
        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        txt = f"{label} {score:.2f}"
        draw.text((x + 3, y + 3), txt, fill="white", font=font)
    if out_path:
        img.save(out_path)
    return img

