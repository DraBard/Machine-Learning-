import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

X = np.loadtxt("toy_data.txt")


###TASK 2


# fig, axes = plt.subplots()
# axes.scatter(X[:, 0], X[:, 1])
# plt.show()

# print(common.init(X, 3, 0))
# print(kmeans.run(X, common.init(X, 1, 0)[0], common.init(X, 1, 0)[1]))

def KMeans_plot():
    for s in range(0, 1):
        for k in range(1, 5):

            Kmeans_mixture = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[0]
            Kmeans_post = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[1]
            print(kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[2])

            common.plot(X, Kmeans_mixture, Kmeans_post, "KMeans")

# print(KMeans_plot())

###TASK 3

# print(naive_em.estep(X, common.init(X, 3, 0)[0]))
# print(naive_em.mstep(X, naive_em.estep(X, common.init(X, 3, 0)[0])[0]))
# print(naive_em.run(X, common.init(X, 3, 0)[0], naive_em.estep(X, common.init(X, 3, 0)[0])))

# print(naive_em.run(X, common.init(X, 3, 0)[0], 0))


###TASK 4

def LL_max():
    l = []
    for s in range(0, 5):
        for k in range(1, 5):
            l.append(naive_em.run(X, common.init(X, k, s)[0], naive_em.estep(X, common.init(X, k, s)[0]))[2])
    for k in range(0, 4):
        print(max(l[k::4]))

# print(LL_max())

def GMM_plot():
    for s in range(0, 1):
        for k in range(1, 5):
            mixture = naive_em.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[0]
            post = naive_em.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[1]
            common.plot(X, mixture, post, "GMM")

# print(GMM_plot())