import argparse
import os

from asd_model.visual_input import *
from asd_model.neural_network import *

# Run from the repository root with: python -m asd_model   (or python main.py; options: --lr, --sessions, --wnt-delta, --ccl5-delta)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test the ASD Wnt/CCL5 neural network model.')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help='learning rate (default 0.03; 0.1 makes the network unstable, see the paper supplement)')
    parser.add_argument('--sessions', type=int, default=100, help='training sessions (passes over the training set)')
    parser.add_argument('--wnt-delta', type=float, default=0.0, help='fold change above normal, ASD = 0.41859')
    parser.add_argument('--ccl5-delta', type=float, default=0.0, help='fold change above normal, ASD = 0.40467')
    args = parser.parse_args()

    # Number of units in the vector made from an image, effectively input layer
    # You can alter this to change the size of the layers in the neural network (units in vector = nodes in layer)
    image_vector_size = 400

    # List of image class categories, names MUST match the names of folders holding the images in image directory
    # You can alter the list items to change the image classes on which the neural network is trained then tested
    class_list = ['trucks', 'dogs', 'cats', 'cars']

    # This is the path to the training image set for the local machine. UPDATE as needed
    train_image_directory = os.getcwd() + "/image_set/train_subset"

    # Dictionary of image class label (key)-list of image vector (value) pairs for training the neural network
    train_image_dict = image2matrix(train_image_directory, class_list, image_vector_size)

    # This is the path to the testing image set for the local machine. UPDATE as needed
    test_image_directory = os.getcwd() + "/image_set/test_subset"

    # Dictionary of image class label (key)-list of image vector (value) pairs for testing the neural network
    test_image_dict = image2matrix(test_image_directory, class_list, image_vector_size)

    # Dictionary of image class label (key)-encoded class label (value) pairs for comparing neural network output
    encoded_label_dict = encode_image_labels(train_image_dict)

    # This is the neural network (an instance of the NeuralNetwork class)
    # You can alter the 'hidden_layer_number', that is the number of layers between the input and output layers
    # You can alter the 'ccl5_delta' for fold differences above/below normal (e.g., 1.5 fold above, enter 0.4)
    # You can alter the 'wnt_delta' for fold differences above/below normal (e.g., 1.5 fold below, enter -0.5)
    net = NeuralNetwork(class_list, image_vector_size, hidden_layer_number=4, ccl5_delta=args.ccl5_delta, wnt_delta=args.wnt_delta)

    # This is the command to train the neural network
    # You can alter the learning rate ('learn_r', --lr) to make the size of weight updates bigger or smaller (default 0.03)
    # You can alter 'sessions' to change how many times the neural network runs the full image set during training
    net.train(train_image_dict, encoded_label_dict, learn_r=args.lr, sessions=args.sessions)

    # This is the command to test the neural network
    # You can set 'use_release_maps' to 'True'/'False' for biological dynamic weighting of signals or fixed weighting
    # Note: Release maps are on for both training and testing (same distribution); fixed weighting is less variable
    net.test(test_image_dict, encoded_label_dict, use_release_maps=True)



