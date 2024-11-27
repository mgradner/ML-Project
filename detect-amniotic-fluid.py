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
    
    # Create train/validation/test splits
    all_images = os.listdir(data_dir)
    train_imgs, test_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.2, random_state=42)
    
    # Assuming you have ground truth labels in a format like:
    # {"image_name.jpg": label}, where label is 0, 1, or 2
    ground_truth = {}  # You'll need to load this from somewhere
    
    # Lists to store predictions and true labels
    y_true = []
    y_pred = []
    
    # Process images and collect predictions
    for image_name in train_imgs:
        image_path = os.path.join(data_dir, image_name)
        
        # Get ground truth label
        # true_label = ground_truth.get(image_name)
        # if true_label is None:
        #     continue
            
        # Get predictions
        results = detector.ensemble_prediction(image_path)
        # pred_label = results['cnn_prediction']
        
        # Store true and predicted labels
        # y_true.append(true_label)
        # y_pred.append(pred_label)
        
        # Plot results
        detector.plot_all_results(image_path, results)
    
    # Calculate and display metrics
    metrics, cm = detector.evaluate_predictions(y_true, y_pred)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")
    
    # Plot confusion matrix
    detector.plot_confusion_matrix(cm)

if __name__ == "__main__":
    main()
