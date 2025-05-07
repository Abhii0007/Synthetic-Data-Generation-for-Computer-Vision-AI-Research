# BlenderMaskGen

**BlenderMaskGen** is a Python-Blender-based pipeline designed for generating high-quality, multiclass segmentation masks using **Object Index** and **Object ID passes**. This synthetic dataset generation approach supports training computer vision models (e.g., semantic and instance segmentation) with pixel-accurate annotations from 3D scenes.

---

## 🔍 Features

- 🎨 **Multiclass segmentation** using Blender’s Object Index pass.
- 🧠 Ideal for deep learning datasets (YOLO, Detectron2, etc.).
- 🧱 Leverages Blender’s powerful rendering pipeline and compositor.
- 📁 Auto-export rendered RGB images, masks, and metadata.
- 🧩 Supports automation for multiple object placements and combinations.

---

## 📷 Sample Outputs

| RGB Image | Segmentation Mask |
|-----------|-------------------|
| ![RGB](samples/rgb_sample.png) | ![Mask](samples/mask_sample.png) |

---

## 🧰 How It Works

1. **Assign Object Index**:
   - In Blender, assign a unique **Pass Index** to each object via the Object Properties panel.

2. **Enable Passes in View Layer**:
   - In `View Layer Properties` → `Passes` tab, enable:
     - **Object Index**
     - (Optionally) **Material Index** or **Cryptomatte** for advanced segmentation.

3. **Compositor Node Setup**:
   - Use Blender’s compositor to separate each object into a binary mask based on their index.
   - Save each mask as a separate `.png` or `.exr` image.

4. **Render and Export**:
   - The script automates camera placement, lighting, object variation, and renders.
   - RGB and mask images are saved in structured folders.

---

## 🧪 Use Cases

- Computer vision dataset creation
- Semantic / instance segmentation
- Object detection model pretraining
- Synthetic data augmentation

---

## 🗂 Folder Structure

