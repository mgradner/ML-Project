# Amniotic Fluid Detection System

## Overview
This system analyzes ultrasound images to detect and classify amniotic fluid levels into three categories:
- Normal
- Low (Oligohydramnios)
- High (Polyhydramnios)

## How It Works

### 1. Image Processing
- Takes ultrasound images as input
- Converts them to grayscale
- Resizes them to a standard size (256x256 pixels)
- Normalizes the pixel values

### 2. Analysis Methods
The system uses three different approaches to analyze each image:

#### A. Convolutional Neural Network (CNN)
- Deep learning model that learns patterns from labeled images
- Classifies images into normal, low, or high fluid levels
- Trained on over 12,000 labeled ultrasound images

#### B. K-means Clustering
- Segments the image into regions
- Helps identify fluid-filled areas
- Provides visual confirmation of fluid regions

#### C. Hierarchical Clustering
- Alternative segmentation method
- Groups similar regions together
- Offers another perspective on fluid distribution

### 3. Results
The system provides:
- Classification result (Normal/Low/High)
- Visualization of fluid regions
- Confidence metrics for the prediction

## Dataset
- Total Images: 12,400
- Distribution:
  - Normal: 40.2% (4,989 images)
  - High: 39.0% (4,838 images)
  - Low: 20.8% (2,573 images)

## Performance
- Uses class weighting to handle imbalanced data
- Includes validation steps to ensure reliability
- Saves results and example predictions for review

## Usage
1. Place ultrasound images in the `data` folder
2. Run the detection script
3. View results in the `test_results` folder
