from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    if label*(np.dot(feature_vector, theta.T) + theta_0) < 1:
        return 1 - label*(np.dot(feature_vector, theta.T) + theta_0)
    else:
        return 0


# feature_vector = np.array([0.34098493, 0.96052049, 0.88206946, 0.15857473, 0.73366259, 0.05654802,
#  0.94979503, 0.74752547, 0.6801385,  0.06514356])
# label, theta, theta_0 = -1, np.array([0.14663404, 0.05205511, 0.05668488, 0.31530876, 0.06815122, 0.88420431,
#  0.05264294, 0.06688735, 0.07351444, 0.76753556]), 0.5

# print( hinge_loss_single(feature_vector, label, theta, theta_0))


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    loss = []
    for i in range(labels.shape[0]):
        if labels[i]*(np.dot(feature_matrix[i,:], theta.T) + theta_0) < 1:
            loss.append(1 - labels[i]*(np.dot(feature_matrix[i,:], theta.T) + theta_0))
        else:
            loss.append(0)
    loss = np.array(loss)
    return np.mean(loss)

# feature_matrix = np.array([[1, 2], [1, 2]])
# labels, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
# print( hinge_loss_full(feature_matrix, labels, theta, theta_0))

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 0:
        return (current_theta + feature_vector*label, current_theta_0 + label)
    else:
        return (current_theta, current_theta_0)

# feature_vector = np.array([-0.22592478, -0.45592498,  0.05456996, -0.30738014,  0.28426198, -0.41012989,
#  -0.2026074,  -0.22451344, -0.45117255, -0.26949194])
# label, theta, theta_0 = 1, np.array([ 0.11426232,  0.18254121,  0.29830345, -0.38839659, -0.22225689, -0.20273161,
#   0.47137551,  0.40055201,  0.43372051, -0.44583106]), 0.2830773541100927
#
# print(perceptron_single_step_update(
#         feature_vector,
#         label,
#         theta,
#         theta_0))

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta = np.zeros(feature_matrix[0, :].shape[0])
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
    return theta, theta_0


# feature_matrix = np.array([[1, 2], [-1, 0]])
# labels, T = np.array([1, 1]), 1
#
# print(perceptron(feature_matrix, labels, T))

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta = average_theta = np.zeros(feature_matrix[0, :].shape[0])
    theta_0 = average_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            average_theta += theta
            average_theta_0 += theta_0
    return average_theta/(feature_matrix.shape[0]*T), average_theta_0/(feature_matrix.shape[0]*T)

# feature_matrix = np.array([[1, 2], [-1, 0]])
# labels, T = np.array([1, 1]), 1
#
# print(average_perceptron(feature_matrix, labels, T))

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lambda value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        theta = (1- eta*L)*current_theta + eta*feature_vector*label
        theta_0 = current_theta_0 + eta*label
        return theta, theta_0
    else:
        theta = (1 - eta * L) * current_theta
        return theta, current_theta_0



def pegasos(feature_matrix, labels, T , L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lambda value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = np.zeros(feature_matrix[0, :].shape[0])
    theta_0 = 0
    j = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            j += 1
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i, :], labels[i], L, 1/j**(1/2), theta, theta_0)
    return theta, theta_0


# feature_matrix = np.array([[1, 1], [1, 1]])
# labels = np.array([1, 1])
# T = 1
# L = 1
# exp_res = (np.array([1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)]), 1)
#
# print(pegasos(feature_matrix, labels, T, L))

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    predicted_labels = []
    for i in range(feature_matrix.shape[0]):
        if np.dot(feature_matrix[i,:], theta) + theta_0 > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(-1)
    return np.array(predicted_labels)

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    classified_train_labels = classify(train_feature_matrix, theta, theta_0)
    classified_val_labels = classify(val_feature_matrix, theta, theta_0)
    compared_train = zip(classified_train_labels, train_labels)
    counter_train = 0
    for i, j in compared_train:
        if i == j:
            counter_train += 1
    compared_val = zip(classified_val_labels, val_labels)
    counter_val = 0
    for i, j in compared_val:
        if i == j:
            counter_val += 1
    return counter_train/len(train_labels), counter_val/len(val_labels)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def stop_words():
    with open('stopwords.txt') as f:
        stop_words = f.readlines()
    return [i.strip("\n") for i in stop_words]


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    stopwords = stop_words()
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)

    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)  #instead of '= word_list.count(word) ' can be just set to 1. With one it is typical bag of words approach when occurance of word is indicated by 1 i a proper place i feature matrix.
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
