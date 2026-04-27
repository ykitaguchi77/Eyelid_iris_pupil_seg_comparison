# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an eye image segmentation project comparing three different methods for segmenting eyelid, iris, and pupil regions in eye images. The project uses U-Net based architectures and includes comprehensive 5-fold cross-validation with resume capabilities.

### Three Methods Being Compared

1. **Method1**: Eyelid segmentation + Ellipse parameter regression for iris/pupil
2. **Method2**: Edge segmentation (3 channels: eyelid edge, iris edge, pupil edge)
3. **Method3**: 6-class region segmentation (background, conjunctiva, visible iris, occluded iris, visible pupil, occluded pupil)

**Current Best Performer**: Method3 with Mean Dice of 0.9424 ± 0.0051

## Key Commands and Workflows

### Running Experiments

```bash
# 1. Data preprocessing (run once)
jupyter notebook process_data.ipynb  # Run All cells in order

# 2. Single-fold training/testing (50 epochs, quick experimentation)
jupyter notebook train.ipynb  # Set flags in cell 7, then Run All

# 3. Full 5-fold cross-validation (300 epochs, production)
jupyter notebook crossvalidation.ipynb  # Run All (supports resume)

# 4. Ablation study with SegFormer
jupyter notebook ablation_SegFormer.ipynb  # Run All (supports resume)

# 5. YOLO11 experiments
jupyter notebook prepare_yolo11_dataset.ipynb  # Prepare YOLO format data
jupyter notebook ablation_yolo11.ipynb  # Run YOLO11 experiments
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install opencv-python numpy pandas scikit-learn scikit-image matplotlib pillow tqdm
pip install transformers  # For SegFormer ablation study
```

### Common Operations in Notebooks

```python
# Resume from interruption (crossvalidation.ipynb)
# Just Run All - automatically resumes from cache/cv_progress.json

# Reset progress and start fresh (crossvalidation.ipynb, cell 9)
RESET_PROGRESS = True  # Set this, run cell, then Run All

# Select which methods to train (train.ipynb, cell 7)
TRAIN_METHOD1 = True
TRAIN_METHOD2 = True
TRAIN_METHOD3 = True

# Windows compatibility (if num_workers errors occur)
NUM_WORKERS = 0  # Set in cell 2 of crossvalidation.ipynb
```

## Architecture and Key Design Patterns

### Data Pipeline Architecture

1. **Raw Data** (CVAT XML annotations) → `process_data.ipynb`
2. **Processed Labels** (Images/labels_seg/, Images/labels_obb/)
3. **6-class unified labels** (sixcls.png files) - Critical for Method3
4. **5-fold split** (fold_indices.json) - Patient-based GroupKFold to prevent data leakage

### Model Training Architecture

All models use:
- **Encoder**: VGG16-BN pretrained backbone
- **Decoder**: U-Net style decoder (64 channels)
- **Heads**: Method-specific output heads

Key differences:
- Method1: Separate regression heads for ellipse parameters
- Method2: 3-channel edge prediction with morphological post-processing
- Method3: 6-class semantic segmentation with ellipse fitting post-processing

### Resume/Progress Management System

The project implements sophisticated resume capabilities:

```python
cache/cv_progress.json = {
    "completed": {"method1_fold0": {...}},  # Completed tasks
    "in_progress": {"method2_fold3": {...}}  # Interrupted tasks
}
```

Progress is saved in real-time after each epoch and fold completion.

### Performance Optimizations

1. **Ellipse parameter caching** (Method1): Pre-compute all ellipse parameters → 25% speedup
2. **Direct sixcls.png loading** (Method3): Skip on-the-fly label generation → 15-20% speedup
3. **Parallel data loading**: num_workers=4 for Linux/Mac (0 for Windows if errors)

## Critical Implementation Details

### Label Generation Logic (Method3 6-class)

The 6-class labeling follows strict logical rules:
```
0: background   = NOT in lid AND NOT in iris AND NOT in pupil
1: conjunctiva  = IN lid AND NOT in iris AND NOT in pupil
2: iris_vis     = IN lid AND IN iris AND NOT in pupil
3: iris_occ     = NOT in lid AND IN iris AND NOT in pupil
4: pupil_vis    = IN lid AND IN iris AND IN pupil
5: pupil_occ    = NOT in lid AND IN iris AND IN pupil
```

### Edge Processing (Method2)

- Training: 3px thick edges with pos_weight=3.0 for class imbalance
- Inference: Morphological closing (25×25 kernel, 6 iterations) fills gaps up to 150px
- Ellipse fitting: Direct cv2.fitEllipse() on clean edges (no RANSAC needed)

### Evaluation Metrics

Unified evaluation across all methods:
- **Eyelid**: Direct mask comparison
- **Iris/Pupil**: Convert to ellipse → mask → Dice coefficient
- All methods evaluated on same ground truth masks

## File Organization

### Core Notebooks
- `process_data.ipynb` - Data preprocessing and label generation
- `train.ipynb` - Single-fold training for quick experiments (50 epochs)
- `crossvalidation.ipynb` - Full 5-fold CV with resume (300 epochs)
- `ablation_SegFormer.ipynb` - SegFormer comparison
- `ablation_yolo11.ipynb` - YOLO11 experiments

### Key Directories
- `Images/images/` - Original images (512×512)
- `Images/labels_seg/` - Segmentation labels including sixcls.png
- `Images/labels_obb/` - Ellipse masks for iris and pupil
- `model/cv_300ep/` - 5-fold CV trained models
- `cache/` - Progress tracking and ellipse parameter cache
- `results/` - CSV evaluation results

### Metadata Files
- `fold_indices.json` - 5-fold patient-based splits
- `image_metadata.csv` - Image metadata with patient IDs
- `patient_list.json` - Unique patient ID list

## Important Notes and Gotchas

1. **Always run process_data.ipynb cells in order** - Dependencies between cells
2. **Large files in .gitignore** - model/ and Images/ directories are not tracked
3. **Windows num_workers issue** - Set NUM_WORKERS=0 if multiprocessing errors occur
4. **Ellipse cache generation** - Run cell 4 in crossvalidation.ipynb before training Method1
5. **Resume works automatically** - Just Run All after interruption, no manual intervention needed
6. **Patient-based splits** - Same patient never appears in both train and val (prevents data leakage)

## Current Experimental State

- **Best Model**: Method3 with Mean Dice 0.9424
- **Latest Results**: See Results.md for detailed performance breakdown
- **Ablation Studies**: SegFormer implemented, YOLO11 in progress
- **All experiments use**: 300 epochs, early stopping at 30 epochs patience

## Development Tips

- Use train.ipynb for quick experiments (50 epochs, single fold)
- Use crossvalidation.ipynb for final evaluation (300 epochs, 5-fold)
- Check improvement.md for implementation history and technical decisions
- Review Experiment.md for detailed experimental setup and rationale