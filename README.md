# Multiclass_Fish_Image_Classification
This project focuses on classifying fish images into multiple categories using deep learning models. The task involves training a Convolutional Neural Network (CNN) from scratch and leveraging transfer learning with pre-trained models to enhance performance. The project demonstrates the complete machine learning pipeline from data preprocessing to model deployment with a Streamlit web application.

#Key Objectives:
Determine the best model architecture for fish image classification
Create a user-friendly web application for real-time predictions
Evaluate and compare metrics across multiple models
Deploy the trained model for practical use
#🎯 Domain & Problem Statement
Domain: Image Classification
Task: Multi-class fish species classification using deep learning

This project addresses the challenge of accurately identifying different fish species from images, which has applications in:

Marine biology research
Fisheries management
Aquatic ecosystem monitoring
Commercial fishing operations
#💡 Skills Developed
Deep Learning: CNN architecture design, transfer learning
Python: TensorFlow/Keras for model development
Data Processing: Image preprocessing, augmentation, normalization
Data Visualization: Matplotlib for training history and results
Model Evaluation: Precision, recall, F1-score, confusion matrix
Deployment: Streamlit for web application development
Version Control: GitHub for code management
#📊 Dataset
Dataset Composition:

Total Images: 8,386 images across 11 fish species
Training Set: 4,984 images
Validation Set: 215 images
Test Set: 3,187 images
Fish Species:
Animal Fish (877 images)
Animal Fish Bass (24 images)
Fish Seafood Black Sea Sprat (456 images)
Fish Seafood Gilted Head Bream (453 images)
Fish Seafood Horse Mackerel (459 images)
Fish Seafood Red Mullet (464 images)
Fish Seafood Red Sea Bream (457 images)
Fish Seafood Sea Bass (431 images)
Fish Seafood Shrimp (461 images)
Fish Seafood Striped Red Mullet (438 images)
Fish Seafood Trout (464 images)
Data Source: Images organized in folders by species, loaded using TensorFlow's ImageDataGenerator

🔧# Methodology & Approach
1. Data Preprocessing & Augmentation
Rescale images to [0, 1] range for normalization
Image resizing to 224×224 pixels (standard size for pre-trained models)
Apply data augmentation techniques:
Rotation (±20°)
Width/Height shift (±20%)
Zoom (±20%)
Horizontal and vertical flipping
Batch size: 32 for efficient processing
2. Model Architecture
Custom CNN Model:
- Conv2D layers: 32, 64, 128 filters with ReLU activation
- Batch Normalization after each convolution
- MaxPooling2D layers for dimensionality reduction
- Dropout (0.5) for regularization
- Dense layers: 256 units with ReLU, output layer with softmax
- Total Parameters: 22,248,395
Transfer Learning Models:
The following pre-trained models from ImageNet were fine-tuned:

VGG16

Pre-trained weights: ImageNet
Fine-tuning: Last 4 layers trainable
Dense layers: 256 units + softmax output
Learning rate: 1e-5
ResNet50

Pre-trained weights: ImageNet
Fine-tuning: Last 10 layers trainable
Global Average Pooling + Dense layers
Learning rate: 1e-5
MobileNet

Lightweight architecture optimized for deployment
Fine-tuning: Last 8 layers trainable
Dense layers: 128 units + softmax
Learning rate: 1e-5
InceptionV3

Pre-trained weights: ImageNet
Fine-tuning: Last 20 layers trainable
Dense layers: 256 units + Dropout(0.4)
Learning rate: 1e-5
EfficientNetV2B0

State-of-the-art efficient architecture
Fine-tuning: Last 20 layers trainable
Dense layers: 256 units + Dropout(0.4)
Learning rate: 1e-5
3. Training Configuration
Hyperparameters:

Optimizer: Adam (learning rate: 1e-5)
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 5-10 with early stopping
Early Stopping: Monitor validation loss, patience=3
Callbacks:

EarlyStopping: Prevent overfitting by monitoring validation loss
ModelCheckpoint: Save best model based on validation accuracy
📈 Model Performance Results
Model	Test Accuracy	Best Validation Accuracy	Key Strength
Custom CNN	38.75%	26.98%	Baseline model
VGG16	95.67%	94.76%	Excellent feature extraction
ResNet50	59.12%	58.94%	Deep architecture
MobileNet	98.05%	97.45%	Best performance
InceptionV3	97.14%	96.37%	High accuracy, well-balanced
EfficientNetV2B0	16.32%	49.15%	Requires more tuning
Best Performing Model: MobileNet
Test Accuracy: 98.05%
Validation Accuracy: 97.45%
Model Size: Lightweight (deployable)
Inference Speed: Fast (suitable for real-time applications)
📊 Evaluation Metrics
Classification Report (MobileNet - Best Model)
                           Precision  Recall  F1-Score  Support
Animal Fish                   0.9755   0.9942   0.9848     520
Animal Fish Bass              0.0000   0.0000   0.0000      13
Black Sea Sprat               0.9609   0.9899   0.9752     298
Gilthead Bream               0.9868   0.9803   0.9836     305
Horse Mackerel               0.9823   0.9720   0.9772     286
Red Mullet                    0.9823   0.9553   0.9686     291
Red Sea Bream                0.9750   1.0000   0.9873     273
Sea Bass                      0.9938   0.9817   0.9877     327
Shrimp                        1.0000   1.0000   1.0000     289
Striped Red Mullet            0.9564   0.9727   0.9645     293
Trout                         0.9966   0.9932   0.9949     292

Overall Accuracy:             0.9805 (3,187 test images)
Macro Average:                0.8918 (Precision), 0.8945 (Recall)
Weighted Average:             0.9767 (Precision), 0.9805 (Accuracy)
🚀 Project Deliverables
1. Trained Models
✅ Custom CNN (.keras & .h5 format)
✅ VGG16 (.keras & .h5 format)
✅ ResNet50 (.keras & .h5 format)
✅ MobileNet (.keras & .h5 format) - Best Model
✅ InceptionV3 (.keras & .h5 format)
✅ EfficientNetV2B0 (.keras & .h5 format)
2. Python Implementation
Complete Jupyter Notebook with all cells executed
Data loading and preprocessing scripts
Model training and evaluation scripts
Prediction and visualization utilities
3. Streamlit Web Application
Interactive web interface for fish image classification
Image upload functionality
Real-time prediction with confidence scores
Model selection dropdown for comparison
Results visualization
4. Documentation
Comprehensive README (this file)
Well-commented code with docstrings
Model comparison report
Training history visualizations
📥 Installation & Setup
Prerequisites
Python 3.8 or higher
pip or conda package manager
CUDA 11.0+ (for GPU acceleration, optional)
Step 1: Clone Repository
git clone https://github.com/Logeswari1931/Multiclass-Fish-Image-Classification.git
cd Multiclass-Fish-Image-Classification
Step 2: Create Virtual Environment
# Using venv
python -m venv fish_env
source fish_env/bin/activate  # On Windows: fish_env\Scripts\activate

# Or using conda
conda create -n fish_classification python=3.10
conda activate fish_classification
Step 3: Install Dependencies
pip install -r requirements.txt
Requirements.txt
tensorflow==2.13.0
keras==2.13.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
scikit-learn==1.3.0
streamlit==1.28.0
opencv-python==4.8.0
Pillow==10.0.0
💻 Usage Instructions
Option 1: Run the Streamlit Web Application
streamlit run app.py
The application will open in your browser at http://localhost:8501

Features:

Upload fish images (JPG, PNG, etc.)
Select model for prediction
View classification results with confidence scores
Compare results across different models
Option 2: Run the Jupyter Notebook
jupyter notebook project.ipynb
Execute cells sequentially to:

Load and explore dataset
Train models (or load pre-trained models)
Evaluate performance
Generate visualizations
Make predictions on test data
Option 3: Use Pre-trained Models in Python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the best model
model = load_model('MobileNetfinal.keras')

# Load and preprocess image
img = image.load_img('path/to/fish.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted Class Index: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
📁 Project Structure
Multiclass-Fish-Image-Classification/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── project.ipynb                          # Main Jupyter Notebook
├── app.py                                 # Streamlit web application
│
├── models/                                # Trained models directory
│   ├── custom_cnn_model.keras
│   ├── VGG16final.keras
│   ├── ResNet50final.keras
│   ├── MobileNetfinal.keras               # Best model
│   ├── InceptionV3final.keras
│   └── EfficientNetV2B0final.keras
│
├── data/                                  # Dataset directory
│   ├── train/                             # Training images
│   ├── val/                               # Validation images
│   └── test/                              # Test images
│
└── utils/                                 # Utility functions
    ├── data_loading.py
    ├── preprocessing.py
    └── evaluation.py
🔬 Model Comparison & Analysis
Key Findings:
Transfer Learning Superiority:

Transfer learning models significantly outperformed the custom CNN
VGG16, MobileNet, and InceptionV3 all achieved >95% accuracy
MobileNet as Optimal Choice:

Highest test accuracy (98.05%)
Smallest model size for deployment
Fastest inference time
Best suited for production environments
Small Class Handling:

Animal Fish Bass (13 images) - challenging class with 0% accuracy
Data augmentation helped but insufficient for very small classes
Future improvement: Collect more samples or use class weights
Trade-offs:

Custom CNN: Fast training, large model size, lower accuracy
VGG16: Good accuracy, requires more computational resources
MobileNet: Best balance of accuracy and efficiency
InceptionV3: High accuracy but slower inference
🎬 Demonstration & Deployment
Live Evaluation
The project includes demo videos and can be evaluated through:

Streamlit Application: Interactive web interface
Jupyter Notebook: Step-by-step execution and visualization
API Endpoints: Can be extended for API integration
LinkedIn Video Demonstration
Link to demo video: [Add your LinkedIn profile link here]

🐛 Known Issues & Limitations
Small Class Problem:

Some fish species have very few training samples (e.g., Animal Fish Bass - 13 images)
Solution: Collect more data or implement class balancing techniques
Model Size:

Some models (VGG16) are quite large (~500MB)
Solution: Use quantization or distillation for deployment
GPU Requirement:

Training is significantly faster with GPU
CPU training can take several hours
🔮 Future Enhancements
Data Collection:

Gather more samples for underrepresented classes
Include various lighting conditions and angles
Advanced Techniques:

Implement ensemble methods combining multiple models
Use focal loss for handling class imbalance
Apply mixup and cutmix augmentation strategies
Deployment:

Convert model to TensorFlow Lite for mobile deployment
Create Docker container for easy deployment
Develop REST API for integration with other applications
Model Optimization:

Quantization for reduced model size
Knowledge distillation from complex to simple models
Neural Architecture Search (NAS) for optimal architecture
Monitoring:

Implement prediction logging and analytics
Track model performance over time
Setup alerts for performance degradation
📚 References & Resources
Frameworks & Libraries:
TensorFlow Documentation
Keras API Reference
Streamlit Documentation
Scikit-learn Documentation
Deep Learning Concepts:
A Guide to Convolutional Neural Networks
Transfer Learning in Deep Learning
ImageNet Classification with Deep Convolutional Neural Networks
Pre-trained Models:
VGG16: Very Deep Convolutional Networks
ResNet50: Deep Residual Learning
MobileNet: Efficient Convolutional Neural Networks
InceptionV3: Rethinking the Inception Architecture
EfficientNet: EfficientNet: Rethinking Model Scaling
✅ Code Standards & Best Practices
PEP 8 Compliance: All code follows Python Enhancement Proposal 8 guidelines
Modular Design: Code organized into reusable functions and classes
Documentation: Comprehensive docstrings for all functions
Error Handling: Proper exception handling throughout the code
Version Control: Regular commits with descriptive messages on GitHub
📄 License
This project is open source and available under the MIT License. Feel free to use, modify, and distribute as per the license terms.

🙏 Acknowledgments
Dataset provided by fish species classification community
TensorFlow and Keras teams for excellent deep learning frameworks
Streamlit for enabling easy web application development
Open source community for various tools and libraries
