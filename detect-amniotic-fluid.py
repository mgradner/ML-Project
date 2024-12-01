import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns

class AmniotiFluidDetector:
    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width
        self.processed_image = None
        self.segmented_image = None
        self.original_shape = None
        self.cnn_model = self.build_cnn()
        
    def build_cnn(self):
        model = tf.keras.Sequential([
            Conv2D(32, 3, activation='relu', input_shape=(self.img_height, self.img_width, 1)),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: normal, low, high fluid
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def load_and_preprocess(self, image):
        # Read image and convert to grayscale
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.original_shape = gray.shape
        
        # Store resized image for CNN
        self.cnn_input = cv2.resize(gray, (self.img_width, self.img_height))
        self.cnn_input = self.cnn_input / 255.0  # Normalize
        self.cnn_input = np.expand_dims(self.cnn_input, axis=-1)
        
        # Reshape for clustering
        self.processed_image = gray.reshape((-1, 1))
        return self.processed_image, self.cnn_input

    def kmeans_clustering(self, n_clusters=3):
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.processed_image)
        
        # Reshape labels back to original image shape
        segmented = labels.reshape(self.original_shape)
        self.segmented_image = segmented
        return segmented
    
    def hierarchical_clustering(self, n_clusters=3):
        # Downsample the image before clustering to reduce memory usage
        max_samples = 10000  # Adjust this number based on your available memory
        
        if len(self.processed_image) > max_samples:
            indices = np.random.choice(len(self.processed_image), max_samples, replace=False)
            sampled_data = self.processed_image[indices]
            
            # Perform clustering on sampled data
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels_sampled = clustering.fit_predict(sampled_data)
            
            # Use nearest neighbors to assign labels to all points
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(sampled_data)
            _, indices = nn.kneighbors(self.processed_image)
            labels = labels_sampled[indices.ravel()]
        else:
            # If data is small enough, proceed with original method
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clustering.fit_predict(self.processed_image)
        
        # Reshape labels back to original image shape
        segmented = labels.reshape(self.original_shape)
        self.segmented_image = segmented
        return segmented

    def ensemble_prediction(self, image_path):
        # Get predictions from all methods
        processed, cnn_input = self.load_and_preprocess(image_path)
        
        # CNN prediction
        cnn_pred = self.cnn_model.predict(np.expand_dims(cnn_input, axis=0))
        
        # Clustering predictions
        kmeans_result = self.kmeans_clustering()
        hierarchical_result = self.hierarchical_clustering()
        
        # Combine predictions (you can modify this based on your needs)
        return {
            'cnn_prediction': np.argmax(cnn_pred[0]),
            'kmeans_segmentation': kmeans_result,
            'hierarchical_segmentation': hierarchical_result
        }

    def plot_all_results(self, image_path, results):
        original = cv2.imread(image_path, 0)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(original, cmap='gray')
        plt.title('Original')
        
        plt.subplot(142)
        plt.imshow(results['kmeans_segmentation'], cmap='nipy_spectral')
        plt.title('K-means Clustering')
        
        plt.subplot(143)
        plt.imshow(results['hierarchical_segmentation'], cmap='nipy_spectral')
        plt.title('Hierarchical Clustering')
        
        plt.subplot(144)
        classes = ['Normal', 'Low', 'High']
        plt.text(0.5, 0.5, f"CNN Prediction:\n{classes[results['cnn_prediction']]}", 
                ha='center', va='center')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def evaluate_predictions(self, y_true, y_pred):
        """
        Calculate and return various performance metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix as a heatmap
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

def main():
    # Setup paths and data loading
    data_dir = "./data"
    
    # Initialize detector
    detector = AmniotiFluidDetector()
    
    # Get all images and their labels from filenames
    all_images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Extract label from filename
            if '_high' in filename:
                label = 2  # high fluid
            elif '_low' in filename:
                label = 1  # low fluid
            elif '_normal' in filename:
                label = 0  # normal fluid
            else:
                continue  # Skip files without labels
                
            all_images.append(filename)
            labels.append(label)
    
    print(f"Found {len(all_images)} labeled images")
    
    # Create train/validation/test splits while preserving label distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Training data preparation
    print("Preparing training data...")
    X_train_data = []
    for image_name in X_train:
        image_path = os.path.join(data_dir, image_name)
        _, cnn_input = detector.load_and_preprocess(image_path)
        X_train_data.append(cnn_input)
    
    X_train_data = np.array(X_train_data)
    y_train = np.array(y_train)
    
    # Validation data preparation
    print("Preparing validation data...")
    X_val_data = []
    for image_name in X_val:
        image_path = os.path.join(data_dir, image_name)
        _, cnn_input = detector.load_and_preprocess(image_path)
        X_val_data.append(cnn_input)
    
    X_val_data = np.array(X_val_data)
    y_val = np.array(y_val)
    
    # Train the CNN model
    print("Training CNN model...")
    history = detector.cnn_model.fit(
        X_train_data,
        y_train,
        validation_data=(X_val_data, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test evaluation
    print("Evaluating model on test set...")
    y_true = []
    y_pred = []
    
    # Create results directory if it doesn't exist
    results_dir = "./test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Process all test images without displaying them
    for image_name, true_label in zip(X_test, y_test):
        image_path = os.path.join(data_dir, image_name)
        results = detector.ensemble_prediction(image_path)
        pred_label = results['cnn_prediction']
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    # Calculate and display metrics
    metrics, cm = detector.evaluate_predictions(y_true, y_pred)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")
    
    # Save metrics to a text file
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write("Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.3f}\n")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # Save random examples
    num_examples = 5  # Adjust this number as needed
    random_indices = np.random.choice(len(X_test), num_examples, replace=False)
    
    print(f"\nSaving {num_examples} random example predictions...")
    with open(os.path.join(results_dir, "example_predictions.txt"), "w") as f:
        for idx in random_indices:
            image_path = os.path.join(data_dir, X_test[idx])
            results = detector.ensemble_prediction(image_path)
            
            # Save the visualization
            plt.figure(figsize=(15, 5))
            detector.plot_all_results(image_path, results)
            example_filename = f"example_{idx}.png"
            plt.savefig(os.path.join(results_dir, example_filename))
            plt.close()
            
            # Log the prediction details
            prediction_text = (
                f"File: {X_test[idx]}\n"
                f"True label: {['Normal', 'Low', 'High'][y_test[idx]]}\n"
                f"Predicted: {['Normal', 'Low', 'High'][results['cnn_prediction']]}\n\n"
            )
            f.write(prediction_text)
            print(prediction_text)

    print(f"Results saved in {results_dir}/")

if __name__ == "__main__":
    main()
