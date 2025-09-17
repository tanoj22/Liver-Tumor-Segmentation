
# Liver Tumor Segmentation from CT Scans using 2D UNet

This project implements a semantic segmentation pipeline to detect liver tumors from grayscale CT scan slices using a custom 2D UNet model trained in PyTorch. It is designed to run efficiently on CPU by training in staged chunks and includes preprocessing, training, validation, and inference.

---

## Project Overview

- **Goal**: Pixel-wise segmentation of liver tumors in 2D CT slices
- **Model**: Custom 2D UNet built from scratch
- **Data**: DICOM-converted 2D grayscale slices (JPG)
- **Loss Function**: Combined Binary Cross-Entropy + Dice loss
- **Training**: Staged chunk-based fine-tuning on 20,000 images (10k tumor + 10k non-tumor)

---

## Key Results

| Metric       | Score   |
|--------------|---------|
| Dice Score   | **0.9022** |
| IoU Score    | **0.8706** |
| Final Model  | `unet_liver_final.pth` |

---

## Notebook Structure

| Notebook | Purpose |
|----------|---------|
| `data_exploration.ipynb`        | Visualize CT and mask samples, inspect pixel ranges and tumor distribution |
| `data_filtering.ipynb`    | Create balanced dataset and chunked training files |
| `dataset_loader.ipynb`      | Define custom PyTorch `Dataset` and visualize batches |
| `unet_training.ipynb` | Trained chunks 1–6 (ended with kernel crash during chunk 6) |
| `unet_training_2.ipynb`   | Resumed from chunk 7, completed training through chunk 8 |
| `validation and inference.ipynb`| Validated on `val_balanced.txt`, computed Dice & IoU, and ran predictions |

---

## Model Architecture

The model is a classic UNet with three encoder–decoder levels:
- Downsampling with `MaxPool2D`, upsampling with `ConvTranspose2D`
- Each block contains two `Conv → BatchNorm → ReLU` layers
- Final output layer applies `sigmoid` to produce a binary segmentation mask

---

## Dataset

- Source: [LiTS Dataset (Kaggle version)](https://www.kaggle.com/datasets/harshwardhanbhangale/lits-dataset)
- Format: 2D grayscale CT slices (`.jpg`)
- Ground truth: Binary masks with `0` = background, `255` = tumor

---

## Training Strategy

Due to memory and CPU constraints, training was performed in **8 chunks** of 2,000 images each.

- `04a_unet_training_chunks1to6.ipynb`: Trained first 6 chunks
- Training crashed during chunk 6 — model was saved as `unet_liver_stage_6.pth`
- `04b_unet_training_resume.ipynb`: Loaded the checkpoint, trained chunks 7 and 8
- Final model saved as: **`unet_liver_final.pth`**

> **Note:** Chunks 9 and 10 were excluded due to a filtering error. Model was finalized after chunk 8 (16,000 training samples).

---

## Sample Predictions

Visual results show CT image, ground truth mask, and predicted mask side-by-side.

Saved in:
```
val_predictions/sample_0.png
val_predictions/sample_1.png
...
```

---

## Inference

To run predictions on new 2D CT images:

```python
python inference.py --image path/to/new_ct.jpg --model unet_liver_final.pth
```

---

## Tech Stack

- Python, PyTorch, OpenCV, NumPy, Matplotlib
- Trained on CPU using staged chunked training
- Jupyter Notebook environment

---

## Final Thoughts

This project demonstrates how to train a fully functional medical segmentation model on a CPU setup using smart resource management and staged fine-tuning. The pipeline is modular and extendable to 3D volumes or other medical domains.

---

## Author

**Sai Tanoj S.**  
M.S. Data Analytics Engineering  
Northeastern University
