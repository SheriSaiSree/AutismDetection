Autism Detection through Image Analysis
A deep learning-based web application for detecting autism using facial image analysis. It utilizes Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs) to enhance training data and improve classification accuracy.

Table of Contents

1.Features
2.Tech Stack
3.Screenshots
4.Installation
5.Usage
6.Model Training
7.Contact
8.ToDo

1.Features
  Real-time image classification via web interface.
  GAN-based data augmentation for robust training.
  Confusion matrix and performance metrics visualization.
  Secure admin interface.
  Dataset loading, preprocessing, and result visualization.

2.Tech Stack
  Frontend: HTML (Django templates)
  Backend: Django, Python
  Deep Learning: TensorFlow, Keras, OpenCV
  Visualization: Matplotlib, Seaborn
  Other Tools: NumPy, Scikit-learn, GAN custom model

3.Screenshots
https://github.com/SheriSaiSree/AutismDetection/tree/main/assets

4.Installation
Clone the repository
cd autism-detection
Set up a virtual environment
Install dependencies

5.Usage
Run the Django server
Open your browser
Login
Username: admin
Password: admin

Steps to Follow:
Load Dataset
Train Model
Test images using GAN + CNN classification
View results and performance metrics

6.Model Training
Dataset is split into 80% training and 20% testing.
CNN is trained using categorical_crossentropy loss and Adam optimizer.
GAN is used to generate augmented facial features for robust classification.

Performance Metrics:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix Visualization

7.Contact
Author: Saisree Sheri
GitHub: https://github.com/SheriSaiSree
Email: sherisaisree@gmail.com

8.TODO / Roadmap
Add mobile responsiveness to frontend
Improve GAN training pipeline
Add model export for real-time inference
