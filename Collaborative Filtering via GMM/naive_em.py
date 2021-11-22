"""Mixture model using EM"""
from typing import Tuple
import numpy as np
import common
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    posteriors = []
    post = []
    for k in range(mixture.p.shape[0]):
        p_posterior = mixture.p[k]/(2*np.pi*mixture.var[k])**(X.shape[1]/2)*np.exp(-np.linalg.norm(X-mixture.mu[k], axis = 1)**2/(2*mixture.var[k]))
        posteriors.append(p_posterior)
    for i in range(len(posteriors)):
        post.append(posteriors[i]/sum(posteriors))
    posteriors_result = np.vstack(post).T
    log_likelihood = np.sum(np.log(np.sum(posteriors, axis = 0)))
    return posteriors_result, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu = []
    var = []
    for k in range(post.shape[1]):
        mu_k = np.sum(post[:, k].reshape(post.shape[0], 1)*X, axis = 0)/np.sum(post[:, k])
        mu.append(mu_k)
    mu = np.vstack(mu)
    p = np.sum(post, axis = 0)/post.shape[0]
    for k in range(post.shape[1]):
        var_k = np.sum(post[:, k]*np.linalg.norm(X - mu[k], axis = 1)**2)/(X.shape[1]*np.sum(post[:, k]))
        var.append(var_k)
    var = np.hstack(var)

    Gauss = GaussianMixture(mu, var, p)

    return Gauss


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    LL_test = [np.sum(np.log(np.sum(post, axis = 0)))]
    i = 0
    while True:
        posterior, LL = estep(X, mixture)
        LL_test.append(LL)
        mixture = mstep(X, posterior)
        i += 1
        if abs(LL_test[i] - LL_test[i-1]) <= 10**(-6)*abs(LL_test[i]):
            break
    return mixture, posterior, LL

