window.onload = function () {
    console.log("✅ annotate.js loaded");

    const img = document.getElementById("annot_img");
    const canvas = document.getElementById("annotationCanvas");
    const saveBtn = document.getElementById("saveBtn");
    const undoBtn = document.getElementById("undoBtn");
    const labelSelect = document.getElementById("labelSelect");
    const ctx = canvas.getContext("2d");
    const clearAllBtn = document.getElementById("clearAllBtn");


    let startX = 0, startY = 0;
    let isDrawing = false;
    // store boxes as {x, y, w, h, category_id}
    window.boxes = [];

    // sync canvas to actual image pixel size
    let scaleX = 1, scaleY = 1;

    function syncCanvasSize() {
        const wrapper = document.getElementById("canvas-container");
        const displayWidth = wrapper.clientWidth;
        const displayHeight = wrapper.clientHeight;

        // Set canvas to display (scaled) size
        canvas.style.width = displayWidth + "px";
        canvas.style.height = displayHeight + "px";

        // Keep its internal resolution to real pixels
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // Calculate scale so mouse coords can map correctly
        scaleX = img.naturalWidth / displayWidth;
        scaleY = img.naturalHeight / displayHeight;

        console.log("Scale:", scaleX, scaleY);
    }


    if (img.complete) {
        syncCanvasSize();
        loadSavedBoxes();
    } else {
        img.onload = function () {
            syncCanvasSize();
            loadSavedBoxes();
        };
    }

    // draw all boxes from window.boxes
    function redrawBoxes() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 2;
        window.boxes.forEach(b => {
            ctx.strokeStyle = "red";
            ctx.strokeRect(b.x, b.y, b.w, b.h);
            // draw label text
            ctx.font = "16px Arial";
            ctx.fillStyle = "red";
            const label = b.category_id ? ("object_" + b.category_id) : "object";
            ctx.fillText(label, b.x + 4, b.y + 16);
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

    const box = {
        x: Math.round(startX),
        y: Math.round(startY),
        w: Math.round(endX - startX),
        h: Math.round(endY - startY),
        category_id
    };

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

}; // window.onload end
