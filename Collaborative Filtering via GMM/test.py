import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# print(X)
# print(X_gold)

mixture, post = common.init(X, K)
post_em, LL = em.estep(X, mixture)

##DEBUGGING
# print(em.estep(X, mixture))
# print(em.mstep(X, post_em, mixture))
print(em.run(X, mixture, post))



