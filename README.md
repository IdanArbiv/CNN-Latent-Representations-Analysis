# CNN-Latent-Representations-Analysis
This project focuses on understanding convolutional neural networks (CNNs) and exploring their latent features. It involves reproducing results from a class tutorial on CIFAR-10 classification, extending the code to include deconvolutional layers for visualizing latent representations, and analyzing the reconstructed images.

# CIFAR-10 Classification
We reproduced the results shown in the class tutorial on CIFAR-10 classification. We followed the tutorial and successfully implemented a basic classification example on the CIFAR-10 dataset, which consists of 32x32 RGB images of various objects. The code provided in the tutorial was used as a starting point, and we made necessary modifications to adapt it to our environment.
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/b518c88e-8bac-4ba8-b210-0097c55df135)

Then we defined our convolution neural network which is
architecture is as follows:
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/b95dbf52-8b8e-45db-b663-823884fa2acf)

The steps we followed to reproduce the results are as follows:

1. Preprocessed the CIFAR-10 dataset by normalizing the image data.
2.Created a CNN model with two convolutional layers, ReLU activation, and max pooling.
3. Trained the model using the training set and evaluated its performance on the test set.
4. Visualized the training and testing accuracy and loss curves.
5. Generated a classification report to measure the overall performance of the model.
6. Attached the reproduced results, including the accuracy and loss curves, and the classification report to the report.

![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/acc32ed1-56b1-429c-9d26-d1427edbfec7)

We chose our loss function to be Cross-Entropy and our optimizer to be SGD.
We trained the model and plot a graph of the loss as a function of epochs for the train and the test datasets and this is the result:
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/8308a5c0-9de5-4f9d-a2e4-0642bb7f777b)
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/e86b99db-9402-4b0c-9da1-5eb2a7e3e9f1)

In our CIFAR-10 classification model, the training loss exhibited a consistent downward trend across iterations, indicating a healthy learning curve and suggesting effective learning from the training data. However, the testing results show considerable variance in class-wise accuracy. The model performed well in distinguishing objects like frogs, cars, ships, and horses but struggled with classes such as cats, birds, and dogs. This disparity might be due to class imbalance or lack of representative features in the training set for these classes. Further work could focus on targeted data augmentation or employing more complex or domain-specific architectures to enhance performance in underperforming classes.



# Deconvolutional Model
We extended the network to include two deconvolutional layers for visualizing latent representations. We modified the network architecture and loss function to incorporate a reconstruction component. The objective was to train the network to not only classify images but also generate reconstructions of the original input.

The goal is to create a pipeline for visualizing the latent representations and generating reconstructed images. By retraining the network with this modified architecture, we can evaluate the classification error on the test set and showcase examples of the original and reconstructed images. This task aims to explore the potential of deconvolutional models for image reconstruction and their impact on classification accuracy.

The steps we followed to implement the deconvolutional model are as follows:

1. Modified the CNN architecture to include two additional deconvolutional layers.
2.Adapted the loss function to include a reconstruction component based on mean squared error (MSE) between the original input and its reconstruction.
3. Trained the modified network using the CIFAR-10 dataset.
4. Evaluated the classification performance on the test set by calculating the accuracy.
5. Generated reconstructed images for a few examples from the test set and compared them with the original images.
6. Attached the classification error (accuracy) and reconstructed images to the report.

We tried several options for the coefficients of the various loss functions using a grid search. We found that for 1 as the coefficient of the cross entropy function and for 2 as the coefficient of the reconstruction function we got the best results.

The classification error on the test set that we got on the new architecture after 20 epochs was 60% which is a little bit better than the previous architecture that we shows on task 1 but not significantly.

We will show now a few examples of original and reconstructed
images after 0, 10, and 20 epochsrespectively.
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/c2cc5a5e-3587-44e7-b97e-3ae16e701eb5)
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/a474aa8e-3f98-4a9b-bf8f-e07cf2dd43c7)
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/475267c4-b2e3-4139-8642-4c1fb6b7d070)

As you can see despite the poor quality of the original images, the
reconstructed images (Especially at the end of training) have been
restored quite well and we are happy with the results. In the
reconstructed images you can see the main patterns that appear in the
original image, although they are slightly more blurred than the original
images.


# Latent Representations Analysis
In this task, we used the trained deconvolutional model to analyze the latent representations. We selected one image from the train set and one from the test set and generated reconstructions for each feature of the latent representations. The aim was to understand the information captured by different channels in the latent space.

We will analyze the latent representations of the trained model from the previous task by generating reconstructions for each feature. We will select one image from the train set and one from the test set, and manipulate the latent representations to examine the impact on the reconstructed images. By zeroing out specific channels in each feature, we can observe the resulting patterns and discuss any meaningful insights that arise. This analysis aims to shed light on the interpretability and significance of different channels in the latent representations of the model, providing a deeper understanding of the learned features.

The steps we followed for the latent representations analysis are as follows:

1. Chose one image from the train set and one from the test set.
2. Generated reconstructions for each feature of the latent representations by selectively setting channels to zero.
3. Repeated the process for all channels in the first convolutional layer and a subset of channels in the second convolutional layer.
4. Attached the reconstructed images to the report and analyzed the patterns and information captured by different channels.

![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/79f94959-179c-4b62-93e0-0f3ab0cbe0b4)
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/5d87a5b3-2e8e-4eee-a72f-df0ccc7de08c)

Upon posing the original image with the six depictions from the first
hidden layer, one can discern the patterns from the original image quite
conspicuously. As a case in point, feature 3 distinctly exhibits the bird's
structure. Feature 5 emphasizes the perch on which the bird is stationed,
while feature 1 brings to light the picture's background. Moreover,
feature 2 captures the bird's color palette and its subtle outline.
Further clarity and depth in the exploration of the original image's
patterns can be found in the sixteen images generated by the second
hidden layer. Features numbered 6-12 provide an enhanced view of the
bird's structure and the perch. The remaining features vividly delineate
the image's background, the bird's color, and various details along the
image's periphery.
These results are heartening from our perspective, suggesting that our
convolutional model is effectively learning and recognizing primary image
patterns, akin to the human visual perception system

![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/3cc62ee8-0a04-418c-a6d6-9c4a8ef8ff1e)
![image](https://github.com/IdanArbiv/CNN-Latent-Representations-Analysis/assets/101040591/5d63edf0-bb3d-428e-91fe-880e85d64e38)

When we look at the original image and the six features from the first hidden layer we clearly see patterns seen in the original image. For example, when looking at features 0 and 3, we see the wings of the plane. In feature 1 we see the body of the plane. In feature 2, the sky color of the background of the image. Also, when we look at the 16 features from the second hidden layer, the different patterns from the original image can be seen even more clearly and deeply. For example,
the structure of the plane and its wings can be identified in features number 1,2 and 8-13 in a much sharper way compared to the previous layer. In feature number 7 we clearly see the color of the blue sky which characterizes the background of the original image. This is very encouraging from our point of view and indicates that even on the test our convolution model learns very well the main patterns of the image that the human eye recognizes.

