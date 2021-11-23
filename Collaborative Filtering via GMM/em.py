"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    posteriors = []
    post = []
    for k in range(mixture.p.shape[0]):
        p_posteriors = []
        for j in range(X.shape[0]):
            x = X[j]
            x_0 = np.where(x != 0)
            SE = np.linalg.norm(x[x_0] - mixture.mu[k][x_0])**2
            p_posterior = np.log((mixture.p[k] + 1e-16)) - len(x_0[0])/2 * np.log(2*np.pi*mixture.var[k]) - SE/(2*mixture.var[k])
            p_posteriors.append(p_posterior)
        p_posteriors = np.array(p_posteriors)
        posteriors.append(p_posteriors)
    for k in range(len(posteriors)):
        fuj = posteriors[k] - logsumexp(posteriors , axis = 0)
        post.append(np.exp(fuj))
    posteriors_result = np.vstack(post).T
    log_likelihood = np.sum(logsumexp(posteriors, axis = 0))
    return posteriors_result, log_likelihood


# post.append((np.exp(posteriors[k])) / (sum(np.exp(posteriors))))


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu = []
    var_k = []
    Cu = []
    #means
    for k in range(post.shape[1]):
        mu_k_nom = []
        mu_k_denom = []
        for n in range(X.shape[0]):
            x = X[n]
            mu_k_nom.append(post[n, k] * x)
            x_0 = np.where(x != 0)
            Cu.append(len(x_0[0]))
            x_copy = x.copy()
            x_copy[x_0] = 1
            mu_k_denom.append(post[n, k] * x_copy)
        mu_k_nom = np.sum(mu_k_nom, axis = 0)
        mu_k_denom = np.sum(mu_k_denom, axis = 0)
        x1 = np.where(mu_k_denom < 1)
        mu_denom = mu_k_nom/mu_k_denom
        mu_denom[x1] = mixture.mu[k][x1]
        mu.append(mu_denom)
    mu = np.vstack(mu)
    #propabilities
    p = np.sum(post, axis = 0)/post.shape[0]
    #variances
    Cu = np.array(Cu[:X.shape[0]]).reshape(-1, 1)
    var_denom = np.sum(post*Cu, axis = 0)
    for k in range(post.shape[1]):
        var_n = 0
        for n in range(X.shape[0]):
            x = X[n]
            x_0 = np.where(x != 0)
            var_n += post[n, k] * np.linalg.norm(x[x_0] - mu[k][x_0])**2
        var_k.append(var_n)
    var_k = np.array(var_k)
    var = var_k/var_denom
    var_min = np.where(var <= min_variance)
    var[var_min] = min_variance

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
        mixture = mstep(X, posterior, mixture)
        i += 1
        if abs(LL_test[i] - LL_test[i-1]) <= 10**(-6)*abs(LL_test[i]):
            break
    return mixture, posterior, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    #estep
    posteriors = []
    post = []
    for k in range(mixture.p.shape[0]):
        p_posteriors = []
        for j in range(X.shape[0]):
            x = X[j]
            x_0 = np.where(x != 0)
            SE = np.linalg.norm(x[x_0] - mixture.mu[k][x_0])**2
            p_posterior = np.log((mixture.p[k] + 1e-16)) - len(x_0[0])/2 * np.log(2*np.pi*mixture.var[k]) - SE/(2*mixture.var[k])
            p_posteriors.append(p_posterior)
        p_posteriors = np.array(p_posteriors)
        posteriors.append(p_posteriors)
    for k in range(len(posteriors)):
        fuj = posteriors[k] - logsumexp(posteriors , axis = 0)
        post.append(np.exp(fuj))
    posteriors_result = np.vstack(post).T

    #Rest of the implementation.
    X_copy = X.copy()
    for n in range(X.shape[0]):
        x_av = []
        x = X[n]
        x_copy = X_copy[n]
        x_0 = np.where(x == 0)
        for k in range(mixture.mu.shape[0]):
            x_av.append(mixture.mu[k][x_0] * posteriors_result[n][k])
        x_av = np.sum(x_av, axis = 0)
        x_copy[x_0] = x_av
    return X_copy
