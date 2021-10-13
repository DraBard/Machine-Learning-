import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # First attempt
    Xn_stack = []
    for n in range(X.shape[0]):
        Xn = []
        sum_exp = 0
        for k in range(theta.shape[0]):
            c = np.amax(np.dot(X[n], theta.T)/temp_parameter)
            sum_exp += np.exp(np.dot(X[n],theta[k].T)/temp_parameter-c)
            Xn.append(np.exp(np.dot(X[n],theta[k].T)/temp_parameter-c))
        Xn_stack.append([1/sum_exp* k for k in Xn])
    H = np.array(Xn_stack).T
    # print(1)
    # print(H)
    return H

    # Increasing efficiency - getting rid of loops
    
    # print(X) 
    # print(theta)

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    # sum_n = 0
    # for n in range(X.shape[0]):
    #     sum_k= 0
    #     for k in range(theta.shape[0]):
    #         log_part = 0
    #         upper_exponent = np.exp(np.dot(X[n],theta[k].T)/temp_parameter)
    #         for l in range(theta.shape[0]):
    #             log_part += np.exp(np.dot(X[n],theta[l].T)/temp_parameter)
    #         if k == Y[n] and upper_exponent/log_part != 0:
    #             sum_k += np.nan_to_num(np.log((upper_exponent/log_part)))
    #     sum_n += sum_k  
    #     sum_k = 0

    #     for k in range(theta.shape[0]):  
    #         sum_d = 0
    #         for d in range(theta.shape[1]):
    #             sum_d += theta[k][d]**2
    #         sum_k += sum_d
    # return -1/X.shape[0]*sum_n+lambda_factor/2*sum_k
    
    
    sum_n = 0
    for n in range(X.shape[0]):
        sum_k= 0
        for k in range(theta.shape[0]):
            log_part = 0
            upper_exponent = np.exp(np.dot(X[n],theta[k].T)/temp_parameter)
            for l in range(theta.shape[0]):
                log_part += np.exp(np.dot(X[n],theta[l].T)/temp_parameter)
            if k == Y[n] and upper_exponent/log_part != 0:
                sum_k += np.nan_to_num(np.log((upper_exponent/log_part)))
        sum_n += sum_k  

    #np.choose for tje second part    
            
    return -1/X.shape[0]*sum_n+lambda_factor/2*sum_k


#stripping co
X = ".1. 79. 32. 37. 49. 76. 18. 29. 36. 63. 21. 1. 25. 86. 79. 37. 91. 60. 18. 85. 79. 60. 1. 74. 36. 13. 11. 59.  5. 25.  1. 37. 50.] 1. 67.  6. 86. 24.  1. 15. 80. 47. 75. 15.] 1. 79.  6. 63. 34. 35. 59. 60. 86. 59. 37.] 1. 85. 32. 57. 75. 70. 98. 50. 66. 48. 18. 1. 32. 70. 64. 89. 98. 86. 81. 68. 32. 40.] 1. 11. 21. 50. 43. 88. 39. 14. 28. 28. 49.] 1. 82. 81. 10. 58.  7. 81. 88. 43. 11. 83.] 1. 74. 70. 37. 47. 54. 30. 35. 92. 32. 95."

X_strip = ""
for i in X:
    if i != "." and i != "]" and i != "[":
        X_strip += i

X_strip = X_strip.split()
X=[]
for i in range(1,11):
    X.append([int(number) for number in X_strip[(i-1)*11:11*i]])

X= np.array(X)

theta = "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]"

theta_strip = ""
for i in theta:
    if i != "." and i != "]" and i != "[":
        theta_strip += i

theta_strip = theta_strip.split()
theta=[]
for i in range(1,11):
    theta.append([int(number) for number in theta_strip[(i-1)*11:11*i]])
theta = np.array(theta)

temp_parameter = 1.0  
lambda_factor = 0.0001

Y = np.array([0, 1,2,3,4,5,6,7,8,9])
 

# print(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
            
            
def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # my code
    # N = X.shape[0]
    # J_gradient = []
    # num_labels = theta.shape[0]      #so for multinomial regression when we classify for 1, 2, 3, 4, 5 it would be 5
    # probability_matrix = compute_probabilities(X, theta, temp_parameter)
    # for m in range(num_labels):
    #     sum_n = np.zeros(X.shape[1])
    #     for n in range(N):
    #         sum_n += X[n]*((1 if Y[n] == m else 0) - probability_matrix[m][n])
    #     J = -1/(temp_parameter*N)*sum_n+lambda_factor*theta[m]
    #     J_gradient.append(J.tolist())
    # J_gradient = np.array(J_gradient)

    # return theta - alpha*J_gradient 
    
    #MIT code
    
    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return train_y % 3, test_y % 3

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    return np.mean(get_classification(X, theta, temp_parameter)%3 == Y)

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
