# Traffic Signs detection using Deep Learning & Image Processing Techniques
This project for an image processing course would involve building a traffic sign detection system using both traditional image processing techniques and deep learning methods. The aim of the project would be to create a system that can accurately detect traffic signs in real-time, using a combination of techniques.

The first step in the project would be to gather a dataset of traffic sign images. This dataset would need to be labeled with information about the type of sign in the image, such as "stop", "yield", or "speed limit". The dataset would be split into a training set and a test set.

The traditional image processing techniques that would be used in the project would include edge detection, which would help to identify the boundaries of the traffic signs in the images. Other techniques such as thresholding and morphological operations would be used to preprocess the images and extract features that would help with classification.

![Filtered image with signs only visible](https://user-images.githubusercontent.com/96639538/229323382-35605d2b-4f5c-40af-be72-732f5c022b3b.png)

![Final output](https://user-images.githubusercontent.com/96639538/229323407-f10d5196-8cb7-47de-90c3-80dc8ed65b69.png)
In this project, we used a dataset of labeled traffic sign images to train a CNN to classify different types of traffic signs. The CNN consisted of several layers, including convolutional layers, pooling layers, and fully connected layers.

The convolutional layers in our CNN used a set of filters to convolve over the input image and produce a set of feature maps. We used the rectified linear unit (ReLU) activation function in our convolutional layers, which helps to introduce non-linearity into the model and allows the network to learn more complex features. ReLU is a commonly used activation function in CNNs because of its computational efficiency and ability to prevent vanishing gradients.

We also included pooling layers in our CNN, which downsample the feature maps produced by the convolutional layers. This helps to reduce the size of the input to the fully connected layers, which can be computationally expensive. We used max pooling in our pooling layers, which takes the maximum value within a pooling window and discards the rest.

![image](https://user-images.githubusercontent.com/96639538/229323589-a0c71312-b344-4875-8003-001cd649138f.png)

![image](https://user-images.githubusercontent.com/96639538/229323595-a2ce8ac3-4656-4a0b-904f-50c4715ffc42.png)

To prevent overfitting, we included dropout layers in our CNN. Dropout randomly drops out a certain percentage of neurons during training, which helps to prevent the network from relying too heavily on specific features in the input. This can improve the generalization of the model and prevent overfitting to the training data.

![image](https://user-images.githubusercontent.com/96639538/229323601-884cbe0a-b1a9-469e-917b-00ab0f3a7505.png)

![image](https://user-images.githubusercontent.com/96639538/229323604-919ae719-a09b-40ee-a9eb-ca9e726f6606.png)


For the final classification layer in our CNN, we used a softmax activation function and cross-entropy loss function. The softmax function outputs a probability distribution over the different classes, which tells us the likelihood of the input image belonging to each class. The cross-entropy loss function compares the predicted probabilities to the actual labels and penalizes the model for incorrect predictions. By minimizing the cross-entropy loss during training, we can optimize the parameters of the CNN to improve its accuracy on the validation set.

![image](https://user-images.githubusercontent.com/96639538/229323613-e970158e-eef4-42b5-ac14-236a561913c4.png)


Overall, this project provided a great opportunity to gain hands-on experience with using CNNs for image classification tasks, as well as exploring different techniques for preventing overfitting and improving the accuracy of the model.
