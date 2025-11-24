# 🏷️ Annotator - AI-Powered Image Annotation Tool

A powerful web-based image annotation tool with integrated machine learning capabilities for object detection and classification. Built with Flask, this tool combines manual annotation with automated detection using Faster R-CNN and Prototypical Networks for few-shot learning.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)

##  Features

###  Manual Annotation
- **Interactive Canvas**: Draw bounding boxes directly on images with an intuitive interface
- **Multi-Class Support**: Annotate objects across 7 predefined categories (Laptop, Monitor, Keyboard, Mouse, Notebook, Neckband, Bottle)
- **Real-time Editing**: Undo, clear, and modify annotations on the fly
- **Navigation**: Seamlessly navigate between images in your dataset
- **COCO Format**: Annotations saved in industry-standard COCO JSON format

###  AI-Powered Detection
- **Faster R-CNN Detector**: Train a custom object detector on your annotated data
- **Prototypical Networks**: Few-shot learning classifier for accurate object classification
- **Auto-Annotation**: Automatically detect and classify objects in new images
- **Model Training**: Train both detector and classifier directly from the web interface
- **Inference Pipeline**: Combined detection and classification for end-to-end automation

###  Data Management
- **COCO Export**: Download annotations in COCO format for use with other tools
- **Annotation Persistence**: All annotations automatically saved to JSON
- **Image Metadata**: Track image dimensions, filenames, and IDs
- **Validation**: Automatic filtering of invalid bounding boxes

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Annotator.git
cd Annotator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask torch torchvision pillow numpy pycocotools
```

3. **Set up your data structure**
```bash
mkdir -p data/images
mkdir -p models
mkdir -p annotations
```

4. **Add your images**
Place your images in the `data/images/` directory.

### Running the Application

1. **Start the Flask server**
```bash
cd webapp
python app.py
```

2. **Access the web interface**
Open your browser and navigate to:
```
http://localhost:5000
```

3. **Start annotating!**
Click "Start Annotating" to begin labeling your images.

##  Usage Guide

### Manual Annotation Workflow

1. **Select an Image**: Navigate to the annotation page (automatically starts at image 0)
2. **Choose a Label**: Select the object category from the dropdown menu
3. **Draw Bounding Boxes**: Click and drag on the image to create bounding boxes
4. **Edit Annotations**: Use the "Undo" button to remove the last box or "Clear All" to start over
5. **Save**: Click "Save" to persist your annotations
6. **Navigate**: Use "Previous" and "Next" buttons to move between images

### Training Machine Learning Models

#### Train the Object Detector

Send a POST request to train the Faster R-CNN detector:

```bash
curl -X POST http://localhost:5000/train_detector \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 40,
    "batch_size": 2,
    "lr": 0.0001
  }'
```

**Parameters:**
- `epochs` (default: 40): Number of training epochs
- `batch_size` (default: 2): Batch size for training
- `lr` (default: 1e-4): Learning rate

#### Train the Classifier

Send a POST request to train the Prototypical Network:

```bash
curl -X POST http://localhost:5000/train_classifier \
  -H "Content-Type: application/json" \
  -d '{
    "n_way": 5,
    "k_shot": 5,
    "q_query": 5,
    "episodes_per_epoch": 100,
    "epochs": 1,
    "lr": 0.0001
  }'
```

**Parameters:**
- `n_way` (default: 5): Number of classes per episode
- `k_shot` (default: 5): Number of support examples per class
- `q_query` (default: 5): Number of query examples per class
- `episodes_per_epoch` (default: 100): Episodes per training epoch
- `epochs` (default: 1): Number of training epochs
- `lr` (default: 1e-4): Learning rate

### Auto-Annotation

Once models are trained, automatically detect objects in an image:

```bash
curl http://localhost:5000/auto_annotate/0
```

This returns detected bounding boxes with:
- `bbox`: [x, y, width, height]
- `score`: Detection confidence
- `category_id`: Predicted class ID
- `category_name`: Predicted class name
- `pred_score`: Classification confidence

##  Project Structure

```
Annotator/
├── webapp/
│   ├── app.py                      # Main Flask application
│   ├── src/
│   │   └── models/
│   │       └── inference.py        # ML model training & inference
│   ├── templates/
│   │   ├── index.html              # Landing page
│   │   └── annotate.html           # Annotation interface
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css           # Styling
│   │   └── js/
│   │       └── annotate.js         # Canvas interaction logic
│   └── annotations/
│       └── instances_default.json  # COCO annotations
├── data/
│   └── images/                     # Your image dataset
├── models/
│   ├── detector.pth                # Trained Faster R-CNN (generated)
│   └── prototypes.pt               # Trained prototypes (generated)
├── annotations/
│   └── instances_default.json      # Annotation file (auto-created)
└── README.md
```

##  API Reference

### Annotation Endpoints

#### `GET /`
Landing page with link to start annotation.

#### `GET /annotate/<image_id>`
Annotation interface for a specific image.

**Parameters:**
- `image_id`: Integer index of the image (0-based)

#### `GET /get_annotations/<image_id>`
Retrieve existing annotations for an image.

**Response:**
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345.0,
      "iscrowd": 0
    }
  ]
}
```

#### `POST /save_annotation`
Save new annotations for an image.

**Request Body:**
```json
{
  "image_id": 0,
  "file_name": "image.jpg",
  "width": 1920,
  "height": 1080,
  "boxes": [
    {
      "x": 100,
      "y": 100,
      "w": 200,
      "h": 150,
      "category_id": 1
    }
  ]
}
```

#### `GET /export_coco`
Download the complete COCO annotations file.

### Machine Learning Endpoints

#### `POST /train_detector`
Train the Faster R-CNN object detector.

**Request Body:**
```json
{
  "epochs": 40,
  "batch_size": 2,
  "lr": 0.0001
}
```

**Response:**
```json
{
  "status": "ok",
  "detector_path": "/path/to/detector.pth"
}
```

#### `POST /train_classifier`
Train the Prototypical Network classifier.

**Request Body:**
```json
{
  "n_way": 5,
  "k_shot": 5,
  "q_query": 5,
  "episodes_per_epoch": 100,
  "epochs": 1,
  "lr": 0.0001
}
```

**Response:**
```json
{
  "status": "ok",
  "proto_path": "/path/to/prototypes.pt"
}
```

#### `GET /auto_annotate/<image_id>`
Run inference on an image using trained models.

**Response:**
```json
{
  "status": "ok",
  "detections": [
    {
      "bbox": [x, y, width, height],
      "score": 0.95,
      "category_id": 1,
      "category_name": "Laptop",
      "pred_score": 0.92
    }
  ]
}
```

##  Supported Object Categories

The tool comes pre-configured with 7 object categories:

| ID | Category Name |
|----|---------------|
| 1  | Laptop        |
| 2  | Monitor       |
| 3  | Keyboard      |
| 4  | Mouse         |
| 5  | Notebook      |
| 6  | Neckband      |
| 7  | Bottle        |

To modify categories, edit the `ensure_annotations_file()` function in [`app.py`](webapp/app.py).

##  Machine Learning Architecture

### Object Detection: Faster R-CNN
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pre-training**: COCO dataset
- **Fine-tuning**: Your custom annotated dataset
- **Output**: Bounding box proposals with confidence scores

### Classification: Prototypical Networks
- **Architecture**: Few-shot learning with metric learning
- **Embedding**: Learned feature representations
- **Similarity**: Euclidean distance to class prototypes
- **Advantage**: Works well with limited training examples

### Inference Pipeline
1. **Detection**: Faster R-CNN proposes object regions
2. **Extraction**: Crop detected regions from image
3. **Classification**: Prototypical Network classifies each region
4. **Post-processing**: Combine detection scores with classification predictions

##  COCO Format Details

Annotations are stored in COCO JSON format with the following structure:

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "image_001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "area": 30000.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "Laptop"
    }
  ]
}
```

##  Development

### Adding New Categories

1. Edit the `ensure_annotations_file()` function in `app.py`:
```python
"categories": [
    {"id": 1, "name": "YourCategory"},
    {"id": 2, "name": "AnotherCategory"},
    # Add more categories...
]
```

2. Update the dropdown in `annotate.html`:
```html
<select id="labelSelect" class="control-select">
    <option value="1">YourCategory</option>
    <option value="2">AnotherCategory</option>
</select>
```

### Customizing Model Parameters

Edit the default parameters in the training endpoints in `app.py`:

```python
epochs = int(body.get('epochs', 40))  # Change default epochs
batch_size = int(body.get('batch_size', 2))  # Change batch size
lr = float(body.get('lr', 1e-4))  # Change learning rate
```

##  Troubleshooting

### Models Not Found Error
**Error**: "Detector or prototypes not found"

**Solution**: Train the models first using the `/train_detector` and `/train_classifier` endpoints.

### Invalid Bounding Box
**Error**: Boxes with width or height < 1 are skipped

**Solution**: Ensure you draw sufficiently large bounding boxes. The system automatically filters out invalid boxes during training.

### Image Not Loading
**Error**: Image fails to display in annotation interface

**Solution**: 
- Verify images are in `data/images/` directory
- Check file permissions
- Ensure image formats are supported (JPG, PNG)

### Training Fails
**Error**: Training endpoint returns error

**Solution**:
- Ensure you have at least a few annotated images
- Check that CUDA is available if using GPU
- Verify annotation file is not corrupted

##  Performance Tips

1. **GPU Acceleration**: Use a CUDA-capable GPU for faster training
2. **Batch Size**: Increase batch size if you have sufficient GPU memory
3. **Image Resolution**: Resize large images to reduce memory usage
4. **Annotation Quality**: More accurate annotations lead to better model performance
5. **Data Augmentation**: The training pipeline includes built-in augmentation

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and utilities
- **COCO**: Annotation format standard

##  Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for efficient image annotation and AI-powered object detection**
