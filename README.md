# pancreas_multitask_nnunet

3D Pancreas CT segmentation and lesion subtype classification using nnU-Net v2 with a custom multi-task ResEncM architecture (segmentation + classification head), following MICCAI reproducibility checklist standards.

---

## Overview
This repository implements a multi-task nnU-Net v2 model for pancreas CT:  
- **Segmentation:** background, pancreas, and lesion  
- **Classification:** lesion subtype (0, 1, 2)  

The model extends the Residual Encoder U-Net (ResEncM) with a custom classification head, aloowing for joint optimization of segmentation and classification tasks.

---

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

**Environment details:**
- OS: Ubuntu 20.04/22.04 (tested)  
- Python: 3.10+  
- GPU: NVIDIA (≥ 12 GB VRAM recommended)  
- CUDA: 12.x (adjust torch build to match)  

---

## Training

prior to  training make sure to:

1. preprocess the data using:
          ```bash
          !nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncM
          ```

2. Generate the trainer_with_classification script
      a. There is a cell in the noteboo which will generate the script and save it to the required location
      b. Dont worry! I will also upload the script here for you to use incase it doesnt work. Simply just make sure it lies on this path:  "..../nnUNet/nnunetv2/training/nnUNetTrainer/trainer_with_classification.py" 


To train the model(s) in this work, run:

```bash
!nnUNetv2_train 501 3d_fullres 0 \
    -p nnUNetResEncUNetMPlans \
    -tr TrainerWithClassification \
    --npz 
```
or 

```bash
!nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr TrainerWithClassification --npz
```

This will start training the model with/using the ResEncM architecture:

- **Architecture:** Residual Encoder UNet (ResEncM)  
- **Tasks:** 3-class segmentation (background, pancreas, lesion) + 3-class classification head (lesion subtype 0/1/2)  
- **Other settings:** Batch size, patch size, learning rate, loss weights, and augmentation are specified in the config file.  

**Example compute:** 1× NVIDIA A100 40GB, training time depends on number of epochs and folds. I used an RTX PRO 6000, it took about 30 minutes for 23 epochs, 0 folds.

> Note: Only fold 0 was trained in the initial implementation due to time constraints.  
> nnUNet allows up to 5-folds, you can specify this in the training command (simply add "-f x", where x is the number of folds).

---

## Inference & Evaluation (No CLI)

**We do NOT use `nnUNetv2_predict` in this repo.**  
Run the notebook cell titled **“nnU‑Net‑style inference”** and update the paths at the top of the cell:

- `MODEL_DIR` → your trained fold (e.g. `/workspace/nnUNet_results/Dataset501_Pancreas/TrainerWithClassification__nnUNetResEncUNetMPlans__3d_fullres/fold_0`)
- `IMAGES_TS` → `/nnUNet_raw/Dataset501_Pancreas/imagesTs`
- `OUT_DIR` → output folder for predictions (e.g. `outputs/segmentation`)

The notebook will:
1) Load the trained ResEncM model (seg + classification head)  
2) Run sliding‑window inference on test images  
3) Save segmentation NIfTI to `OUT_DIR`  
4) Write lesion subtype predictions to:
```
outputs/classification_predictions.csv
```

**Metrics (aligned with *Metrics Reloaded* recommendations):**
- **Segmentation:** Dice Similarity Coefficient (DSC, overlap)  
- **Classification:** macro/weighted F1 

---

- **ResEncM Fold-0 Model** 
  Trained on Dataset501 Pancreas CT data, using nnU-Net v2 with classification head.  

This model produces:
- Pancreas DSC ≈ 0.84  
- Lesion DSC ≈ 0.53  
- Subtype macro-F1 ≈ 0.48  

---

## Results

The model achieves the following performance on the pancreas dataset:

| Model name       | Pancreas DSC | Lesion DSC | Subtype Macro-F1 | Balanced Accuracy |
|------------------|--------------|------------|------------------|-------------------|
| ResEncM (fold_0) | 0.84         | 0.53       | 0.48             | 0.41              |

A visual example (GT vs Prediction overlays) is available in `results_examples/`.

---
