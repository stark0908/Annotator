window.onload = function () {
    console.log("✅ annotate.js loaded");

    const img = document.getElementById("annot_img");
    const canvas = document.getElementById("annotationCanvas");
    const saveBtn = document.getElementById("saveBtn");
    const undoBtn = document.getElementById("undoBtn");
    const labelSelect = document.getElementById("labelSelect");
    const ctx = canvas.getContext("2d");
    const clearAllBtn = document.getElementById("clearAllBtn");
    const trainBtn = document.getElementById("trainBtn");
    const autoDetectBtn = document.getElementById("autoDetectBtn");

    let startX = 0, startY = 0;
    let isDrawing = false;
    // store boxes as {x, y, w, h, category_id}
    window.boxes = [];

    // sync canvas to actual image pixel size
    let scaleX = 1, scaleY = 1;

    function syncCanvasSize() {
    const img = document.getElementById("annot_img");
    const canvas = document.getElementById("annotationCanvas");

    // Get displayed dimensions of the image
    const rect = img.getBoundingClientRect();
    const displayWidth = rect.width;
    const displayHeight = rect.height;

    // Set canvas display size to match image exactly
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";

    // Set internal resolution
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Scale mouse positions correctly
    scaleX = img.naturalWidth / displayWidth;
    scaleY = img.naturalHeight / displayHeight;

    console.log("Scale:", scaleX, scaleY);
}




    if (img.complete) {
        syncCanvasSize();
        loadSavedBoxes();
        autoDetectBtn.disabled = false; // ALWAYS allow auto-detection

    } else {
        img.onload = function () {
            syncCanvasSize();
            loadSavedBoxes();
            autoDetectBtn.disabled = false; // ALWAYS allow auto-detection

        };
    }

    // draw all boxes from window.boxes
    function redrawBoxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    window.boxes = window.boxes.filter(b => {
        let x = b.x, y = b.y, w = b.w, h = b.h;

        // ✅ Ensure positive width/height
        if (w < 0) { x = x + w; w = Math.abs(w); }
        if (h < 0) { y = y + h; h = Math.abs(h); }

        // ✅ Skip invalid boxes
        if (w < 5 || h < 5) {
            console.warn("⚠️ Skipping invalid box:", b);
            return false;
        }

        // ✅ Draw safely
        ctx.strokeStyle = "red";
        ctx.strokeRect(x, y, w, h);

        ctx.font = "16px Arial";
        ctx.fillStyle = "red";
        const label = categoryNames[b.category_id] || ("class_" + b.category_id);
        ctx.fillText(label, x + 4, y + 16);

        // ✅ Update box in storage (fix negative)
        b.x = x; b.y = y; b.w = w; b.h = h;
        return true;
    });
}


    // Load existing annotations for this image
    function loadSavedBoxes() {
        const imageId = parseInt(imageIdFromTemplate);
        fetch(`/get_annotations/${imageId}`)
            .then(res => res.json())
            .then(data => {
                if (data && data.annotations) {
                    // convert COCO format annotations -> our box objects
                    window.boxes = data.annotations.map(a => {
                        const [x, y, w, h] = a.bbox;
                        return { x: x, y: y, w: w, h: h, category_id: a.category_id };
                    });
                    console.log("Loaded boxes:", window.boxes);
                    redrawBoxes();
                }
            })
            .catch(err => {
                console.warn("No saved boxes / error loading:", err);
            });
    }

    // Mouse handling (drawing)
    canvas.addEventListener("mousedown", function (e) {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        // convert client coords into canvas-space (account for CSS scaling)
        startX = (e.clientX - rect.left) * scaleX;
        startY = (e.clientY - rect.top) * scaleY;
    });

    canvas.addEventListener("mousemove", function (e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    redrawBoxes();

    ctx.strokeStyle = "orange";
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, mouseX - startX, mouseY - startY);
    });


    canvas.addEventListener("mouseup", function (e) {
    if (!isDrawing) return;
    isDrawing = false;

    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;

    const category_id = parseInt(labelSelect.value) || 1;

    // Normalize box to ensure positive width/height
    let x1 = Math.round(startX);
    let y1 = Math.round(startY);
    let x2 = Math.round(endX);
    let y2 = Math.round(endY);

    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);

    // Prevent invalid small/zero boxes
    if (w < 5 || h < 5) {
        console.log("Ignored tiny/invalid box");
        return;
    }

    const box = { x, y, w, h, category_id };

    window.boxes.push(box);
    console.log("Box added:", box);

    redrawBoxes();
    });



    // Undo last box
    undoBtn.addEventListener("click", function () {
        if (window.boxes.length > 0) {
            const removed = window.boxes.pop();
            console.log("Removed box:", removed);
            redrawBoxes();
        }
    });


        // Clear All Annotations
    clearAllBtn.addEventListener("click", function () {
        if (window.boxes.length > 0) {
            if (confirm("Are you sure you want to remove all annotations?")) {
                window.boxes = [];
                console.log("All boxes cleared");
                redrawBoxes();
            }
        } else {
            alert("No annotations to clear.");
        }
    });


    // Save annotation: include image metadata (file_name, width, height)
    saveBtn.addEventListener("click", function () {
        console.log("Saving...", window.boxes);
        const imageId = parseInt(imageIdFromTemplate);
        // Get file name from imagePathFromTemplate (strip leading path if present)
        // imagePathFromTemplate is like "/static_images/image_0001.png"
        let fileName = imagePathFromTemplate.split("/").pop();

        // send image pixel dims too (naturalWidth/naturalHeight)
        const payload = {
            image_id: imageId,
            file_name: fileName,
            width: img.naturalWidth,
            height: img.naturalHeight,
            boxes: window.boxes
        };

        fetch("/save_annotation", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            console.log("Saved:", data);
            alert("Annotations saved!");
        })
        .catch(err => {
            console.error("Save failed:", err);
            alert("Save failed: " + err);
        });
    });

    trainBtn.addEventListener("click", async () => {
    if (!confirm("Train models using your saved annotations?")) return;

    trainBtn.disabled = true;
    trainBtn.textContent = "Training...";

    try {
        // Step 1: Train Detector
        let detRes = await fetch("/train_detector", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})  // empty JSON since no params were passed
        }).then(r => r.json());
        if (detRes.status !== "ok") throw new Error(detRes.error || "Detector training failed");

        // Step 2: Train Classifier
        let clsRes = await fetch("/train_classifier", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
        }).then(r => r.json());
        if (clsRes.status !== "ok") throw new Error(clsRes.error || "Classifier training failed");

        alert("✅ Training complete!");
        autoDetectBtn.disabled = false; // enable auto detect button

    } catch (err) {
        alert("Training failed: " + err);
    }

    trainBtn.textContent = "Train Models";
    trainBtn.disabled = false;
    });


    autoDetectBtn.addEventListener("click", async () => {
        const imageId = parseInt(imageIdFromTemplate);

        autoDetectBtn.disabled = true;
        autoDetectBtn.textContent = "Detecting...";

        try {
            let res = await fetch(`/auto_annotate/${imageId}`).then(r => r.json());
            if (res.status !== "ok") throw new Error(res.error);

            // Convert detections to window.boxes format
            const exists = (b1, b2) => (
                b1.x === b2.x && b1.y === b2.y && b1.w === b2.w && b1.h === b2.h
            );

            res.detections.forEach(det => {
                let [x, y, w, h] = det.bbox;

                // ✅ Clamp to positive
                if (w < 0) { x = x + w; w = Math.abs(w); }
                if (h < 0) { y = y + h; h = Math.abs(h); }

                w = Math.max(5, w);
                h = Math.max(5, h);

                const newBox = { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h), category_id: det.category_id };

                if (!window.boxes.some(b => exists(b, newBox))) {
                    window.boxes.push(newBox);
                }
            });

            console.log("Auto detections added:", window.boxes);
            redrawBoxes();  // call your existing redraw function

        } catch (err) {
            alert("Auto detection failed: " + err);
        }

        autoDetectBtn.textContent = "Auto Detect";
        autoDetectBtn.disabled = false;
    });

}; // window.onload end



