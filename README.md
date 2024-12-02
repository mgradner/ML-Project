# Amniotic Fluid Detection System

## Overview
This system analyzes ultrasound images to detect and classify amniotic fluid levels into three categories:
- Normal (40-60% fluid content)
- Low/Oligohydramnios (<40% fluid content)
- High/Polyhydramnios (>60% fluid content)

## System Components

### 1. Image Preprocessing (highlow.py)
This script performs initial image categorization:
- Analyzes grayscale intensity of ultrasound images
- Calculates fluid percentage based on dark pixel areas
- Automatically labels images based on fluid content:
  - `_high` suffix: ≥60% fluid
  - `_normal` suffix: 40-60% fluid
  - `_low` suffix: <40% fluid

### 2. Deep Learning Analysis (detect-amniotic-fluid.py)
The main detection system uses three different approaches:

#### A. Image Processing
- Converts images to grayscale
- Resizes to 256x256 pixels
- Normalizes pixel values

#### B. Analysis Methods
1. **Convolutional Neural Network (CNN)**
   - Deep learning model for classification
   - Trained on 12,400 labeled images
   - Uses class weighting to handle data imbalance

2. **K-means Clustering**
   - Segments fluid regions
   - Provides visual confirmation

3. **Hierarchical Clustering**
   - Alternative segmentation approach
   - Used for detailed analysis on sample images

## Dataset
- Total Images: 12,400
- Distribution:
  - Normal: 40.2% (4,989 images)
  - High: 39.0% (4,838 images)
  - Low: 20.8% (2,573 images)

## Model Performance
- Training accuracy: ~90%
- Validation accuracy: ~77%
- Class weighting applied:
  - Higher weight (1.61) for low fluid cases
  - Lower weights (~0.84) for normal and high cases
- Early stopping to prevent overfitting

## Results Output
The system generates a `test_results` folder containing:
- `metrics.txt`: Overall performance metrics
- `confusion_matrix.png`: Visual representation of model accuracy
- Example predictions with:
  - Original image
  - Segmentation visualizations
  - CNN classification results

## Usage
1. **Initial Setup**:   ```bash
   # Run initial categorization
   python highlow.py   ```

2. **Main Analysis**:   ```bash
   # Run detection system
   python detect-amniotic-fluid.py   ```

3. **View Results**:
   - Check `./test_results/` folder for:
     - Performance metrics
     - Example predictions
     - Visualization plots
