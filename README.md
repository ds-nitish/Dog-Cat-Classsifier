# Dog-Cat Image Classifier with MobileNet

This repository contains code for a dog-cat image classifier built using the MobileNet architecture. The classifier is trained to distinguish between images of dogs and cats, providing a binary classification.

## MobileNet

MobileNet is a lightweight deep learning architecture designed specifically for mobile and embedded devices. It offers a good balance between accuracy and computational efficiency, making it suitable for resource-constrained environments. MobileNet achieves this efficiency by utilizing depthwise separable convolutions, which significantly reduce the number of parameters and computational cost compared to traditional convolutional layers.

## Dataset

The dog-cat image classifier is trained on a labeled dataset of dog and cat images. The dataset consists of a collection of images, with each image labeled as either a dog or a cat. It is important to have a diverse and balanced dataset to ensure the classifier's ability to generalize well to unseen data. The images in the dataset are preprocessed by resizing them to a common size (e.g., 224x224 pixels) and normalizing the pixel values to be within a certain range (e.g., [0, 1]).

## Usage

To use the dog-cat image classifier, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/your-username/dog-cat-classifier.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:

   - Organize your dog and cat images into separate folders, ensuring that each folder contains images corresponding to its respective class.
   - Split your dataset into training and testing sets, placing the images into appropriate folders. For example:
     - Training Set:
       - `data/train/dog`: Folder containing dog images for training
       - `data/train/cat`: Folder containing cat images for training
     - Testing Set:
       - `data/test/dog`: Folder containing dog images for testing
       - `data/test/cat`: Folder containing cat images for testing

4. Training the Classifier:

   - Open the `train.py` file and modify the necessary parameters according to your requirements. This includes parameters such as the batch size, number of epochs, learning rate, and model saving location.
   - Run the training script:

     ```
     python train.py
     ```

   The script will train the classifier using the provided dataset. The MobileNet model will be trained on the dog and cat images, adjusting its parameters to minimize the classification error.

5. Evaluating the Classifier:

   - After training, the model will be saved with its weights.
   - Use the evaluation script to assess the performance of the classifier on the test set:

     ```
     python evaluate.py
     ```

   The script will load the trained model and evaluate its accuracy on the test set. It will provide metrics such as accuracy, precision, recall, and F1-score, giving you an understanding of the classifier's performance.

6. Making Predictions:

   - Use the trained model to make predictions on new images using the prediction script:

     ```
     python predict.py path/to/image.jpg
     ```

   Replace `path/to/image.jpg` with the path to the image you want to classify. The script will load the trained model, preprocess the image, and output the predicted class label (dog or cat) along with the corresponding confidence score.

## Contributing

Contributions to improve the dog-cat image classifier or extend it to other image classification tasks are welcome. If you encounter any issues or have suggestions, please open an issue or submit a pull request. Your contributions can include adding new features, optimizing the code, enhancing the dataset, or improving the model's performance.


