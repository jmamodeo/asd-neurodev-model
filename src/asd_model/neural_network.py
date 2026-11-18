import numpy as np

from asd_model.network_tuning import *
from asd_model.loss_outputs import *

# Default learning rate. A learning rate of 0.1 makes the N(0,1)-initialised network unstable (shuffled training,
# softmax output, release map in backprop), so 0.03 is the default (selected from a control-only pilot, see
# docs/paper and README). The larger value, 0.1, is kept as ORIGINAL_LEARNING_RATE and can be chosen with
# learn_r=ORIGINAL_LEARNING_RATE (or --lr 0.1 in main.py / reproduce.py); the paper's supplement uses it.
DEFAULT_LEARNING_RATE = 0.03
ORIGINAL_LEARNING_RATE = 0.1


# Below is the Neural Network class with its functions for neural network creation, training, and testing.

class NeuralNetwork:
    """
    This class creates instances of a neural network for classifying images. The layer size,
    the number of hidden layers, and the classes used for image classification are
    customizable.

    Upon instance creation, the class initiates layer sizes, weights, and biases.
    This class has a 'train' method for teaching network instances which kinds of images belong to
    which image classes. It has a 'test' method for testing the accuracy of a trained neural network
    instance in classifying new images (of the same kind used in training) with the learned image
    classes.

    This class has a 'feedforward' method for converting images to image class probabilities used to make
    classification predictions, and it has a 'feedback' method used to update layer weights and biases
    during training based on the calculated loss between the prediction and the true class label
    of the image.

    The activity of neural network instances of this class is modulated by 2 mathematical formulations
    of 2 signaling molecules: ccl5 and wnt. In biological neural networks, these govern the release of
    the excitatory neurotransmitter glutamate and the stability of synapses between neurons, respectively.
    These signaling molecules are passed into neural network instances as 'deltas', differences between
    the normal levels and those found in the brain of ASD patients. Both 'ccl5_delta' and 'wnt_delta'
    default to 0 to simulate normal brain activity.
    """

    def __init__(self, class_list, image_vector_size, hidden_layer_number, ccl5_delta=0, wnt_delta=0):
        """
        This function initializes neural network layer sizes, weights, and biases.

        It takes a class list, used to set the size of the output layer for comparison to the encoded
        labels of images, which also equal class list length. It takes an image vector size, which serves
        as the input layer size and is used to set the sizes of the hidden layers. It takes a hidden layer
        number, used to create a matching number of hidden layers in the neural network, equal in size
        to the input layer (i.e., image vector size).

        All hidden layers start equal in size to the input layer to simulate the sparse encoding of visual signals
        seen in the visual pathway of humans and other primates. That is to say, each neuron (node) in a
        visual brain region or regional subdivision (layer) holds a unique section (pixel) of the visual
        scene (image), represented as an activity/firing pattern (number). As information is passed from
        one layer to the next, each neuron (node) conveys its activity (number) to a downstream neuron
        one neuron that will primarily represent its unique section (pixel) of the visual scene and to one
        neuron representing every other unique section in the visual scene (image). This is an idealized
        unique 1-to-unique 1 relationship preventing the transfer of redundant information, which is the
        posited goal of spare encoding in biological neural networks.

        Wnt supports synaptic stability and higher/lower levels have been shown to lead to a higher/lower
        number of synapses formed between a given neuron and its downstream partners. In this function,
        'wnt_delta' stores a fold increase (e.g, 1.4 found in the brain of ASD patients) or decrease
        compared to normal wnt levels. This value is used as a scaling factor for layer size: it is
        multiplied by the size of the upstream layer to get the number of additional nodes that will be
        added or subtracted from the size of the upstream layer to get the size of the downstream layer.
        Scaling up creates redundancy, where some neurons in the upstream layer convey their activity to
        more than one neuron that will, in turn, primarily represent their unique section of the visual
        scene (i.e., multiple nodes representing the same pixel). Whereas, scaling down makes encoding
        even sparser: several neurons in the upstream layer converge on a single neuron in the downstream
        layer (i.e., representations of multiple pixels are pooled into the same node).

        Here, the 'ccl5_delta' value is stored for later use in get_release_maps(), see network_tuning.py.

        Parameters
        ----------
        class_list
        image_vector_size
        hidden_layer_number
        ccl5_delta
        wnt_delta
        """

        # Initializing layer sizes
        self.layer_sizes = [image_vector_size]
        for layer in range(hidden_layer_number):
            additional_nodes = int(self.layer_sizes[-1] * wnt_delta)
            layer_size = self.layer_sizes[-1] + additional_nodes
            self.layer_sizes.append(layer_size)
        self.layer_sizes.append(len(class_list))

        # Initializing layer weights, biases
        self.layer_weights = []
        self.layer_biases = []
        for layer in range(hidden_layer_number + 1):
            layer_weights = np.random.randn(self.layer_sizes[layer + 1], self.layer_sizes[layer])
            self.layer_weights.append(layer_weights)
            layer_biases = np.random.randn(self.layer_sizes[layer + 1], 1)
            self.layer_biases.append(layer_biases)

        # Initializing layer modulation
        self.ccl5_delta = ccl5_delta

        # Release-map sampler (replaceable by an external harness) and the last forward-pass state
        self.release_map_fn = get_release_maps
        self.last_release_maps = None
        self.last_effective_weights = None

    def feedforward(self, image_vector, use_release_maps=True):
        """
        This function passes the pixel values of the image vector (input layer) through
        the hidden layers to the output layer, with the inputs from one layer to the next
        scaled by distinct weight matrices, and return a list of layer activities.

        A sigmoid function is applied to the activity calculated for each hidden layer, and
        a softmax functon is applied to the activity calculated for the output layer. These
        are important for comparing the output of the network to the encoded label of images,
        see network_tuning.py.

        The use of release maps simulating stochastic neurotransmitter release for dynamic
        weighting vs fixed weighting of synaptic inputs (see network_tuning.py) can be
        turned on/off by setting 'use_release_maps' to True/False.

        The output layer uses softmax (the last weight matrix is detected by its index in the loop below).

        The release maps sampled here are stored in 'self.last_release_maps' (one per weight
        matrix) and the effective weights E = W * M in 'self.last_effective_weights', so that
        feedback() can backpropagate through exactly the weights used in the forward pass. If
        'use_release_maps' is False, M is all ones. Maps are cast to the weight dtype.

        Parameters
        ----------
        image_vector
        use_release_maps

        Returns
        -------
        layer_activities
        """

        # Keep the sampled release maps (and effective weights) so backprop uses the same
        # weights as the forward pass
        if use_release_maps:
            sampled_maps = self.release_map_fn(self.layer_weights, self.ccl5_delta)
            release_maps = [np.asarray(m, dtype=w.dtype) for m, w in zip(sampled_maps, self.layer_weights)]
        else:
            release_maps = [np.ones_like(w) for w in self.layer_weights]
        effective_weights = [w * m for w, m in zip(self.layer_weights, release_maps)]
        self.last_release_maps = release_maps
        self.last_effective_weights = effective_weights

        # Calculating layer activities
        layer_activities = [image_vector]
        for layer in range(len(self.layer_weights)):
            layer_activity = np.dot(effective_weights[layer], layer_activities[-1]) + self.layer_biases[layer]
            # The layer index stops at len - 1, so the last weight matrix (the output layer) is the one
            # that gets the softmax; every hidden layer gets a sigmoid.
            if layer == len(self.layer_weights) - 1:
                normalized_activity = apply_softmax(layer_activity)
            else:
                normalized_activity = apply_sigmoid(layer_activity)
            layer_activities.append(normalized_activity)

        return layer_activities

    def feedback(self, encoded_label, layer_activities, learn_r, release_maps=None, max_gradient_norm=1.0):
        """
        This function updates the weights and biases for connections between layers of the
        network based on the differences between image class probabilities held by nodes in
        the output layer and the encoded image label.

        It takes an encoded image label and subtracts it from the output layer activity produced
        by the paired image, grabbed from a list of layer activities also passed in, to get a delta
        (difference) matrix. The values of this matrix are propagated backwards by multiplying it
        by the transposition of the weight matrix the output layer and the last hidden layer. These
        values are then multiplied by the result of [the activity of the last hidden layer * (1 - the
        activity of the last hidden layer)] to get a delta matrix for the last hidden layer (reflecting
        the contribution each of its node make to the differences in the output layer vs the encoded
        label). These delta values are back-propagated to each hidden layer using the same strategy.

        The deltas for each layer are clipped by first making them into one long vector (1D matrix),
        adding the square of each value up, and taking the square root of the sum for the Euclidean
        norm (single value). If the Euclidean norm is greater than a pre-set max gradient norm value,
        a clipping factor is calculated by dividing the max gradient norm by the Euclidean norm. The
        deltas for each layer are then multiplied by this clipping factor, making them smaller and
        easier to handle for backpropagation.

        The deltas for each layer used to update their associated weights by multiplying the layer delta
        by the transposition of the layer activity. The values in the resulting matrix are scaled by
        multiplying it by a pre-set learning rate ('learn_r') passed in. This scaled matrix of weight
        updates is then subtracted from the existing layer weight matrix.

        Layer deltas are also used to update the biases associated with each layer by first multiplying
        the layer delta values by the learning rate (to scale them), then subtracting the existing layer
        biases matrix by the scaled layer delta matrix.

        Backpropagation goes through the EFFECTIVE weights E = W * M used in the forward pass
        (M = release maps, 'release_maps' argument or 'self.last_release_maps' if None): the delta of the
        previous layer is (E.T @ delta) * a * (1 - a), and the weight gradient is (delta @ a_prev.T) * M
        (chain rule through W * M). Bias gradients are unchanged. 'max_gradient_norm' (default 1.0) is
        the global clipping threshold.

        Because the output layer is a softmax, output_delta = p - label is exactly the gradient of the
        cross-entropy loss with respect to the output pre-activations.

        Cross-entropy loss (difference) of neural network output layer activity vs encoded image label
        is calculated by taking the natural log of node values (the smaller the numerical input, the
        larger the negative numerical output), which is the inverse of Euler's number**node value,
        applied to layer activities via the Softmax function during feedforward(). The loss matrix,
        equal in size to the output layer, is multiplied by the encoded image label to zero out
        values at all nodes representing incorrect image classes and preserving the value at the node
        representing the correct/true image class. This matrix is summed and converted to a positive
        value before being returned.

        Parameters
        ----------
        encoded_label
        layer_activities
        learn_r
        release_maps
        max_gradient_norm

        Returns
        -------
        loss
        """

        # Use the release maps from the forward pass
        # (when the maps are the ones from the immediately preceding feedforward, W has not changed since,
        # so the stored effective weights are reused instead of recomputing W * M)
        if release_maps is None:
            release_maps = self.last_release_maps
            effective_weights = self.last_effective_weights
        else:
            effective_weights = None
        if release_maps is None:
            release_maps = [np.ones_like(w) for w in self.layer_weights]
        if effective_weights is None:
            effective_weights = [w * m for w, m in zip(self.layer_weights, release_maps)]

        # Calculating differences in output matrix vs label matrix
        output_delta = layer_activities[-1] - encoded_label

        # Calculating loss based on output vs label
        loss = -np.sum(encoded_label * np.log(layer_activities[-1] + 1e-10))

        # Backpropagation of differences in output matrix vs label matrix
        # Propagate through the effective weights E = W * M, and use 1 - a directly
        # (no float64 np.ones temporary that would upcast float32 activities).
        layer_deltas = [output_delta]
        for layer in range(len(self.layer_weights) - 1):
            back_layer = -1 * (layer + 1)
            delta = np.dot(effective_weights[back_layer].T, layer_deltas[-1])
            activity = layer_activities[back_layer - 1]
            layer_delta = delta * (activity * (1 - activity))
            layer_deltas.append(layer_delta)

        # Clipping gradients to reduce exploding or vanishing weights
        all_gradients = np.concatenate([delta.flatten() for delta in layer_deltas])
        gradient_norm = np.linalg.norm(all_gradients)

        if gradient_norm > max_gradient_norm:
            clipping_factor = max_gradient_norm / gradient_norm
            for index in range(len(layer_deltas)):
                layer_deltas[index] = layer_deltas[index] * clipping_factor

        # Updating layer weights, biases
        for layer in range(len(self.layer_weights)):
            back_layer = -1 * (layer + 1)
            # Chain rule through W * M gives dL/dW = (delta @ a_prev.T) * M
            weight_gradient = np.dot(layer_deltas[layer], layer_activities[back_layer - 1].T) * release_maps[back_layer]
            self.layer_weights[back_layer] -= learn_r * weight_gradient
            self.layer_biases[back_layer] -= learn_r * layer_deltas[layer]

        return loss

    def run_session(self, training_image_dict, encoded_label_dict, learn_r=DEFAULT_LEARNING_RATE):
        """
        This function runs one training session: one pass over every training image, in a freshly
        shuffled order, with a forward pass (release maps on) and a weight update per image.

        The (image class, image vector) pairs of all classes are pooled and shuffled with
        np.random.permutation, so images from different classes are interleaved within a session
        (updates are not blocked by class and a session does not end on a single class).

        Returns
        -------
        losses (per example, in presentation order), class_names (in presentation order)
        """

        # Shuffled presentation order across all classes
        examples = [(image_class, image_vector)
                    for image_class, image_list in training_image_dict.items()
                    for image_vector in image_list]

        losses = []
        class_names = []
        for index in np.random.permutation(len(examples)):
            image_class, image_vector = examples[index]
            layer_activities = self.feedforward(image_vector, use_release_maps=True)
            loss = self.feedback(encoded_label_dict[image_class], layer_activities, learn_r)
            losses.append(loss)
            class_names.append(image_class)

        return losses, class_names

    def train(self, training_image_dict, encoded_label_dict, learn_r=DEFAULT_LEARNING_RATE, sessions=100):
        """
        This function trains the neural network to classify images according to the specific class
        labels with which the images are paired.

        Predictions of image class are made using feedforward(), and weights applied
        to the values passed between layers to make predictions are updated using feedback().

        This function takes a session count and iterates over it for a specified number of training sessions.
        It also takes a dictionary of image class name (key)-list of image vectors (value) pairs curated
        for training and for each session, each image class key and its paired list of image vectors
        is pulled out. The image vectors are individually passed into feedforward(), see above,
        and a list of layer activities is returned.

        This function takes an encoded label dictionary with image class name(key)-encoded class label
        pairs. The current image class is given as the key to this dictionary to get the encoded
        label for the image class. The list of layer activities and the encoded class label for
        the current image class is passed into feedback(), see above, along with a pre-set
        learning rate given as an argument to this train() function, and returns loss. The loss
        returned for each image is added to a total loss, with which the average loss for each
        session is calculated and printed, along with the session number.

        Each session is one shuffled pass over all images (run_session()), and the printed
        session loss is the mean loss per training example (total loss / number of training images).

        Loss values for each session are also stored in a list with its corresponding session number
        stored in a sessions number list. The loss over time list is used in conjunction with the
        list of session numbers to plot mean loss over sessions with plot_loss_over_time(),see
        loss_outputs.py. This plot is rendered after training is complete.

        The loss values for each image class are stored in a class losses list. After training on an
        image class, the associated class losses list is added to a loss table dictionary, with the
        image class as the key and the class losses as the value. After training on all image classes,
        the loss table dictionary is added to a loss table list. This is combined with the session
        number list mentioned above to create a loss dataframe that is exported to Excel.

        Parameters
        ----------
        training_image_dict
        encoded_label_dict
        learn_r
        sessions
        """

        number_of_images = sum(len(image_list) for image_list in training_image_dict.values())

        loss_over_time = []
        loss_table_list = []
        session_list = []
        for session in range(sessions):
            losses, class_names = self.run_session(training_image_dict, encoded_label_dict, learn_r)
            total_loss = sum(losses)

            loss_table = {image_class: [] for image_class in training_image_dict.keys()}
            for loss, image_class in zip(losses, class_names):
                loss_table[image_class].append(loss)
            loss_table_list.append(loss_table)

            loss_over_time.extend(losses)
            session_list.extend([session + 1] * len(losses))

            # Mean loss per training example
            avg_loss = total_loss / number_of_images
            print(f"Session: {session + 1} of {sessions}, Loss: {avg_loss:.4f}")

        print("Training complete")
        plot_loss_over_time(loss_over_time, session_list)
        export_loss_table(loss_table_list, session_list)

    def evaluate(self, image_dict, encoded_label_dict, use_release_maps=True):
        """
        This function measures performance without updating any weights or biases. It returns the
        mean cross-entropy loss per example (-sum(label * log(output + 1e-10)), the same definition
        used in feedback()) and the accuracy percentage (argmax of the output layer vs argmax of
        the encoded label).

        Returns
        -------
        mean_loss_per_example, accuracy_percent
        """

        total_loss = 0.0
        num_correct = 0
        num_images = 0
        for image_class, image_list in image_dict.items():
            encoded_label = encoded_label_dict[image_class]
            label = int(np.argmax(encoded_label))
            for image_vector in image_list:
                output = self.feedforward(image_vector, use_release_maps)[-1]
                total_loss += -np.sum(encoded_label * np.log(output + 1e-10))
                num_correct += int(int(np.argmax(output)) == label)
                num_images += 1

        return total_loss / num_images, (num_correct / num_images) * 100

    def test(self, test_image_dict, encoded_label_dict, use_release_maps=True):
        """
        This function tests the neural network on classifying new image (distinct for those in the
        training image set) according to the image classes the network learned in training.

        It takes a dictionary of image class name (key)-list of image vectors (value) pairs curated
        for testing. Each image class key and its paired list of image vectors is pulled out and the
        image vectors are individually passed into feedforward(), see above, and a list of layer
        activities is returned. The index of the node with the highest activity value/image class
        probability in the OUTPUT layer is the network's prediction.

        The prediction is int(argmax(layer_activities[-1])) (the output layer, not the input image),
        compared with == to int(argmax(encoded_label)). The accuracy percentage is printed and also
        returned.

        'use_release_maps' defaults to True so testing uses the same release-map distribution as
        training.

        This function takes an encoded label dictionary with image class name(key)-encoded class label
        pairs. The current image class is given as the key to this dictionary to get the encoded
        label for the image class.

        If the prediction and the label match, the trial is considered correct and a 1 is added to the
        total correct trial count. After all images have been tested, the total correct trial count is
        divided by the total number of images tested (times 100) to get a prediction accuracy
        percentage, which is printed and returned.

        Parameters
        ----------
        test_image_dict
        encoded_label_dict
        use_release_maps

        Returns
        -------
        accuracy
        """

        num_correct = 0

        for image_class in test_image_dict.keys():
            image_list = test_image_dict[image_class]
            encoded_label = encoded_label_dict[image_class]
            # Compare the argmax of the network output with the argmax of the label
            label = int(np.argmax(encoded_label))

            for image_vector in image_list:
                layer_activities = self.feedforward(image_vector, use_release_maps)
                prediction = int(np.argmax(layer_activities[-1]))

                if prediction == label:
                    num_correct += 1

        all_images = [len(x) for x in test_image_dict.values()]
        accuracy = (num_correct / sum(all_images)) * 100

        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
