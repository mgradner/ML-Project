import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns

class AmniotiFluidDetector:
    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width
        self.cnn_model = self.build_cnn()
        
    def build_cnn(self):
        model = tf.keras.Sequential([
            Conv2D(32, 3, activation='relu', input_shape=(self.img_height, self.img_width, 1)),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def load_and_preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img / 255.0
        return np.expand_dims(img, axis=-1)

def main():
    print("Starting Amniotic Fluid Detection System")
    
    # Setup
    data_dir = "./data"
    detector = AmniotiFluidDetector()
    
    # Load data
    print("\nLoading and preprocessing images...")
    all_images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            if '_high' in filename:
                label = 2
            elif '_low' in filename:
                label = 1
            elif '_normal' in filename:
                label = 0
            else:
                continue
            all_images.append(filename)
            labels.append(label)
    
    # Data distribution
    print("\nDataset distribution:")
    for label, name in enumerate(['Normal', 'Low', 'High']):
        count = labels.count(label)
        print(f"{name}: {count} images ({count/len(labels)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Prepare training data
    print("\nPreparing training data...")
    X_train_data = [detector.load_and_preprocess(os.path.join(data_dir, img)) for img in X_train]
    X_train_data = np.array(X_train_data)
    y_train = np.array(y_train)
    
    # Train model
    print("\nTraining model...")
    history = detector.cnn_model.fit(
        X_train_data, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    y_true = []
    y_pred = []
    
    # Process test set
    for image_name in X_test:
        img = detector.load_and_preprocess(os.path.join(data_dir, image_name))
        pred = detector.cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)
        y_pred.append(np.argmax(pred[0]))
    y_true = y_test
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Save results
    print("\nSaving results...")
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            line = f"{metric.capitalize()}: {value:.3f}"
            print(line)
            f.write(line + '\n')
    
    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"\nResults saved in {results_dir}/")

if __name__ == "__main__":
    main()
