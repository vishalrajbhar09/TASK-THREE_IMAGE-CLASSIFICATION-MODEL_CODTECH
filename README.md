INTERN VISHAL RAMKUMAR RAJBHAR

Intern ID:- CT4MESG

Domain:- Machine Learning

Duration:-December 17, 2024, to April 17, 2025

Company:- CODETECH IT SOLUTIONS

Mentor:- Neela Santhosh Kumar




Task Title: Image Classification Model - Build a Convolutional Neural Network (CNN)

This repository contains the implementation of a Convolutional Neural Network (CNN) designed to classify images from the CIFAR-10 dataset. The project highlights the steps involved in preparing image data, designing an efficient CNN architecture, and evaluating the model's performance using TensorFlow/Keras.

The CIFAR-10 dataset is a widely used benchmark dataset comprising 60,000 32x32 color images in 10 classes. In this project, a CNN is trained to classify these images into their respective categories, such as airplanes, automobiles, birds, and more. The CNN architecture leverages convolutional layers for feature extraction, max-pooling layers for down-sampling, and fully connected layers for final classification.

Key Deliverables:

•	Model Performance: The CNN achieves high accuracy on the test dataset, showcasing its ability to effectively classify images.

•	Training and Validation Trends: Visualizations of accuracy and loss over epochs provide insights into the learning process and model optimization.

•	Data Insights: A detailed exploration of the dataset with sample images and corresponding labels highlights the diversity of CIFAR-10.

This project focuses on creating a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset. The project demonstrates how to effectively implement, train, and evaluate a CNN while providing insights into its performance through visualizations.


Key Features:


•	Dataset Preparation:

o	The CIFAR-10 dataset, consisting of 60,000 color images across 10 categories (Airplane, Automobile, Bird, etc.), is used.

o	Images are normalized to the range [0, 1] to improve model training performance.

o	Labels are one-hot encoded to facilitate multi-class classification.

OUTPUT:-01
![image](https://github.com/user-attachments/assets/37665ed9-0edc-4931-ab27-6b8e0e6e7133)



•	Model Architecture:

o	The CNN is built with three convolutional layers using the ReLU activation function and MaxPooling layers to down-sample feature maps.

o	A flattening layer converts the output into a vector, which is fed into a dense (fully connected) layer for feature extraction.

o	The output layer uses a softmax activation function to classify images into one of 10 categories.



•	Model Compilation and Training:

o	The model is compiled with the Adam optimizer, categorical crossentropy loss function, and accuracy as the performance metric.

o	Training is performed over 10 epochs with a batch size of 64, and validation is conducted on the test dataset.


OUTPUT>-02
![image](https://github.com/user-attachments/assets/88a2d172-7ef2-4afe-bb56-647e96ad7e7b)


•	Performance Evaluation:

o	The model's performance is evaluated using the accuracy metric on the test dataset, achieving competitive results.

o	Accuracy and loss trends are plotted over epochs for both training and validation datasets to analyze convergence and potential overfitting.

OUTPUT OF MODEL ACCURACY AND LOSS:- 03
![image](https://github.com/user-attachments/assets/45fe4f75-7d8c-41c9-951f-8382ee26c6c9)
![image](https://github.com/user-attachments/assets/857091a2-5d85-47b4-a23c-d83469427a46)



•	Visualizations:

o	A sample of training images is displayed along with their respective class labels for dataset exploration.

o	Training and validation accuracy/loss curves are plotted to provide insights into the learning process.
________________________________________


This project is a demonstration of how CNNs can be used for image classification tasks and serves as a foundational example for machine learning enthusiasts exploring deep learning. Contributions, feedback, and collaborations are always welcome. Feel free to explore the repository and dive into the code!

