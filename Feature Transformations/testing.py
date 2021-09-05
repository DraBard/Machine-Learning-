# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 01:11:47 2021

@author: palusbar
"""

import pdb
import numpy as np
import feature_transformation_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features2 = [('cylinders', hw3.one_hot,),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)


if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------


# score_perceptron = hw3.xval_learning_alg(hw3.perceptron, auto_data, auto_labels, 10)
# print(score_perceptron)

# score_averaged_perceptron = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data, auto_labels, 10)
# print(score_averaged_perceptron)


#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# score_averaged_perceptron2 = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10)
# print(score_averaged_perceptron2)

# print(dictionary)
# print(review_texts)
# print(review_labels)
# print(review_bow_data)
# print(hw3.reverse_dict(dictionary))

list1 = []
for i in range(len(review_bow_data)):
    word = np.sum(review_bow_data[i])
    list1.append(word)
  
sorted_list = sorted(list1, key=int, reverse = True)
# print(sorted_list)

#look for 10 words most used in positive reviews
positive_ten = []
for i in range(len(list1)):
    
    if review_labels[0][list1.index(sorted_list[i])] == 1:
        positive_ten.append(sorted_list[i])
    if len(positive_ten) == 10:
        break
    
print(positive_ten)

#look for 10 words most used in positive reviews
negative_ten = []
for i in range(len(list1)):
    
    if review_labels[0][list1.index(sorted_list[i])] == -1:
        negative_ten.append(sorted_list[i])
    if len(negative_ten) == 10:
        break
    
print(negative_ten)

reversed_dictionary = hw3.reverse_dict(dictionary)

#look for 10 positive words that occurs most often


positive_ten_words = []
for i in range(len(positive_ten)):
    print(list1.index(positive_ten[i]))
    print(reversed_dictionary[list1.index(positive_ten[i])])
    positive_ten_words.append(reversed_dictionary[list1.index(positive_ten[i])])

print(positive_ten_words)

reversed_dictionary = hw3.reverse_dict(dictionary)

#look for 10 negative words that occurs most often


negative_ten_words = []
for i in range(len(negative_ten)):
    print(list1.index(negative_ten[i]))
    print(reversed_dictionary[list1.index(negative_ten[i])])
    negative_ten_words.append(reversed_dictionary[list1.index(negative_ten[i])])
print(negative_ten_words)



#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    raise Exception("implement me!")

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    raise Exception("modify me!")


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    raise Exception("modify me!")


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    raise Exception("modify me!")

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

