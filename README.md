# For-rest-from-Fires-Fire-Detection-Model-using-Deep-Learning-
## Project Description
This initiative focuses on creating a Convolutional Neural Network (CNN) model to identify fire in images. The goal is to develop an effective tool for early fire detection, applicable in various fields such as surveillance systems, forest monitoring, and safety mechanisms.

## Problem Statement
Early fire detection is vital for averting disasters and saving lives. Conventional fire detection methods, like smoke detectors and human observation, can be slow and ineffective in certain situations. This project utilizes deep learning techniques to build a CNN model that can automatically detect fire, smoke, and non-fire in images, offering a quicker and more dependable solution.

## Data Sources
The dataset for this project comprises images categorized as ‘fire’, ‘smoke’, and ‘non-fire’.

## Methodology
1. Data Extraction:
Zip File Handling: The dataset, stored in a zip file, is extracted to a designated directory using the zipfile module.
Directory Structure: The dataset is arranged into training and testing directories, facilitating the data loading process.
2. Data Augmentation:
ImageDataGenerator: The ImageDataGenerator class from tensorflow.keras.preprocessing.image is employed for data augmentation, applying random transformations such as rotation, width/height shifts, shear, zoom, and horizontal flip to the training set to improve model generalization.
3. Model Architecture:
Pre-trained Model: A pre-trained MobileNetV2 model (with ImageNet weights) serves as the base model, utilizing transfer learning.
Custom Layers: The model incorporates a GlobalAveragePooling2D layer, a Dense layer with 128 units and ReLU activation, and a final Dense layer with 3 units and softmax activation for classification.
4. Learning Rate Scheduler:
LearningRateScheduler Callback: A custom learning rate scheduler is employed to reduce the learning rate after a specified number of epochs, aiding in fine-tuning the model and mitigating overfitting.

## Model Training and Evaluation
Model Compilation: The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
Training: The model is trained over several epochs for demonstration purposes. In a practical scenario, more epochs would likely be necessary for optimal performance.
Evaluation: The model’s effectiveness is measured on the test set, providing metrics such as loss and accuracy.

## Data Preprocessing
Resizing: All images are resized to a consistent dimension to be compatible with the CNN.
Normalization: Pixel values are scaled to a range between 0 and 1.
Data Augmentation: Techniques like rotation, flipping, and zooming are applied to increase the variety within the training dataset.

## Saving the Model
The trained model is saved to disk, allowing for reuse or deployment without the need for retraining.

## Prediction and Visualization
Image Preprocessing: User-provided images are loaded, resized, and pre-processed before being input into the model for prediction.
Result Display: The predicted class labels are shown alongside the input images using matplotlib.
Confusion Matrix and Accuracy: A confusion matrix is created to visualize the model’s performance across different classes, and the overall accuracy score is calculated and displayed.

## Dependencies
The project necessitates the following dependencies:
Python 3.x
Google Colab
TensorFlow
NumPy
Pandas
Matplotlib
scikit-learn

   
