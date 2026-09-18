## 🌸 Polycystic Ovary Syndrome (PCOS) Detection Using Ultrasound Images
This repository contains an end-to-end deep learning workflow implemented in TensorFlow and Keras to detect Polycystic Ovary Syndrome (PCOS) from ovarian ultrasound images. The pipeline experiments with 5 distinct custom Convolutional Neural Network (CNN) architectures, handles rigorous medical data augmentation, visualizes internal performance curves, and exports the top-performing model for raw inference diagnostics.
The script follows a structured workflow divided into data ingestion, exploratory visualization, augmentation, experimental architecture evaluation, and deployment.


## 🛠️ Code Architecture & Core Workflow
## 1. Library Ecosystem & Imports 📚
* Purpose: Sets up the development workspace.
* Details: Imports TensorFlow/Keras layers for deep modeling, alongside standard computer vision (cv2), data matrix handling (numpy, pandas), and graph plotting (matplotlib) libraries.

## 2. Exploratory Dataset Loading & Visualization 🔍
* Purpose: Initial structural inspection of raw images.
* Details: Streams raw imagery straight from the filesystem, extracts underlying class directories dynamically, and plots an exploratory grid of labeled 224x224 ultrasound scans to verify image integrity before downstream ingestion.

## 3. Medical Image Augmentation & Generators 🧪
* Purpose: Mitigates medical image overfitting and handles data splitting.
* Details: Normalizes pixel matrix distributions (1./255) and injects real-world scan variations (rotations, zooms, horizontal/vertical flips). Splitting parameters route 70% of directories to train_it and 30% to val_it.

## 4. Empirical Model Architecture Variations (Models 1–5) 🏗️
The repository tests five custom Sequential configurations to find the optimal layer depths and kernel dimensions for diagnostic feature extraction:
* Model 1 (2-Conv Layers, 5 epochs): Uses 5×5 filters downsampled by aggressive 4×4 pooling layers to capture baseline structural properties.
* Model 2 (🏆 Selected Final Model, 3-Conv Layers, 8 epochs): Extends spatial exploration with sequential 6×6, 5×5, and 3×3 convolutional blocks paired with shrinking pooling pools (6×6 → 5×5 → 3×3).
* Model 3 (3-Conv Layers, 6 epochs): Tightens internal feature representations using fewer target filter kernels (10 → 12 → 5) across an accelerated training timeline.
* Model 4 (3-Conv Layers, 10 epochs): A deeper variant trained across a longer baseline timeline to evaluate convergence thresholds.
* Model 5 (3-Conv Layers, 7 epochs): A balanced approach mixing intermediate kernel grids (15 → 12 → 8) with customized pooling strategies.

## 5. Categorical Loss Compilation & History Tracking 📈
* Purpose: Standardizes training optimization metrics.
* Details: Leverages the Adam optimizer alongside a two-node categorical cross-entropy loss function. After training execution completes, validation and training loss arrays are instantly plotted to catch signs of early overfitting.

## 6. Model Serialization & Inference Deployment 🚀
* Purpose: Exports the production weights and runs diagnostic verification.
* Details: Saves Model 2 into a self-contained HDF5 (.h5) format. The loading engine takes an unseen test ultrasound frame (img1.jpg), mirrors training conditions precisely (rescaling and expanding data dimensions to a 4D batch array), passes it to the network layer, and creates a clean dictionary lookup mapping diagnostic trust values to class definitions (infected vs notinfected).

## 🧬 Deep Learning Pipeline Phases
   📂 Data Ingestion ──> 🛠️ Custom CNN ──> ⚙️ Optimization ──> 🩺 Automated Inference
   (Preprocessing)       (Architecture)       (Training)          (Diagnostics)

## 🔹 1. Data Ingestion & Image Preprocessing
The system uses TensorFlow's data utilities (ImageDataGenerator and image_dataset_from_directory) to automatically load, label, and batch images from organized local folders. It scales the image dimensions specifically to 224x224 pixels with 3 color channels (RGB) to prepare them for neural network processing.
## 🔹 2. Custom Convolutional Neural Network (CNN) Architecture
Instead of using a heavy pre-trained model, the project builds a lightweight, high-efficiency custom sequential CNN. It consists of:
* Three Convolutional Layers (Conv2D): These layers act as automated feature extractors, scanning the images to detect edges, textures, shapes, and anomalies.
* Three Max Pooling Layers (MaxPooling2D): These layers drastically reduce the spatial dimensions of the data after each convolution, keeping only the most prominent features. This prevents overfitting and speeds up calculation.
* A Flatten and Softmax Dense Layer: The final layers take the 2D feature maps, flatten them into a 1D vector, and pass them to a 2-node output layer. The softmax activation outputs a probability distribution between the two target classes (Infected vs. Healthy).

## 🔹 3. Model Optimization & Training
The network is configured using the Adam optimizer and evaluates its errors via Categorical Crossentropy loss. It tracks both training accuracy and validation accuracy. The training performance (loss and accuracy curves) is plotted and visualised using Matplotlib.
## 🔹 4. Post-Processing & Automated Diagnostics (Inference)
Once trained, the model takes an unseen image and outputs a raw prediction array containing probability scores. The code automatically targets the maximum probability (prediction.max()), translates that mathematical index into a human-readable text label using a mapping function (get_key()), and flags the result—in this case, correctly outputting a diagnosis of "infected".


## 📂 Repository Directory Structure
Ensure your local data paths align perfectly with the target paths hardcoded in the scripts:

├── input/
│   └── pcos-detection-using-ultrasound-images/
│       └── data/
│           ├── train/
│           │   ├── infected/
│           │   └── notinfected/
│           └── test/
│               └── infected/
│                   └── img1.jpg
└── pcos_detection_pipeline.py

------------------------------
## ⚡ Setup & Installation
Install the verified baseline dependencies using standard terminal compilation:
pip install tensorflow opencv-python numpy pandas matplotlib

