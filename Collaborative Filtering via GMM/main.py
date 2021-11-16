import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt

X = np.loadtxt("toy_data.txt")

# fig, axes = plt.subplots()
# axes.scatter(X[:, 0], X[:, 1])
# plt.show()

# print(common.init(X, 1, 0))
# print(kmeans.run(X, common.init(X, 1, 0)[0], common.init(X, 1, 0)[1]))

for s in range(0, 5):
    for k in range(1, 5):

        Kmeans_mixture = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[0]
        Kmeans_post = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[1]
        print(kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[2])

        common.plot(X, Kmeans_mixture, Kmeans_post, "KMeans")