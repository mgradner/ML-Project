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
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    y_true = []
    y_pred = []
    
    # Create figure for sample predictions
    plt.figure(figsize=(15, 10))
    sample_size = min(5, len(X_test))  # Show up to 5 examples per class
    
    # Dictionary to store examples by class
    class_examples = {0: [], 1: [], 2: []}
    
    # Process test set and collect examples
    for idx, image_name in enumerate(X_test):
        img = detector.load_and_preprocess(os.path.join(data_dir, image_name))
        pred = detector.cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_class = np.argmax(pred[0])
        true_class = y_test[idx]
        
        y_pred.append(pred_class)
        
        # Store example if we need more for this class
        if len(class_examples[true_class]) < sample_size:
            class_examples[true_class].append({
                'image': img,
                'pred': pred_class,
                'name': image_name
            })
    
    y_true = y_test
    
    # Plot sample predictions
    class_names = ['Normal', 'Low', 'High']
    fig, axes = plt.subplots(3, sample_size, figsize=(15, 10))
    
    for class_idx in range(3):
        for sample_idx in range(min(sample_size, len(class_examples[class_idx]))):
            example = class_examples[class_idx][sample_idx]
            ax = axes[class_idx, sample_idx]
            ax.imshow(example['image'].squeeze(), cmap='gray')
            ax.axis('off')
            color = 'green' if example['pred'] == class_idx else 'red'
            ax.set_title(f'True: {class_names[class_idx]}\nPred: {class_names[example["pred"]]}', 
                        color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'sample_predictions.png'))
    plt.close()

    # Calculate and save distribution of test set
    test_distribution = {
        'Normal': y_test.count(0),
        'Low': y_test.count(1),
        'High': y_test.count(2)
    }
    
    # Plot test set distribution
    plt.figure(figsize=(8, 6))
    plt.bar(test_distribution.keys(), test_distribution.values())
    plt.title('Test Set Distribution')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(results_dir, 'test_distribution.png'))
    plt.close()
    
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
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()
    
    print(f"\nResults saved in {results_dir}/")

if __name__ == "__main__":
    main()
