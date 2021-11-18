import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

X = np.loadtxt("toy_data.txt")

# fig, axes = plt.subplots()
# axes.scatter(X[:, 0], X[:, 1])
# plt.show()

# print(common.init(X, 3, 0))
# print(kmeans.run(X, common.init(X, 1, 0)[0], common.init(X, 1, 0)[1]))

# for s in range(0, 5):
#     for k in range(1, 5):
#
#         Kmeans_mixture = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[0]
#         Kmeans_post = kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[1]
#         print(kmeans.run(X, common.init(X, k, s)[0], common.init(X, k, s)[1])[2])
#
#         common.plot(X, Kmeans_mixture, Kmeans_post, "KMeans")

# print(naive_em.estep(X, common.init(X, 3, 0)[0]))
# print(naive_em.mstep(X, naive_em.estep(X, common.init(X, 3, 0)[0])[0]))
print(naive_em.run(X, common.init(X, 3, 0)[0], naive_em.estep(X, common.init(X, 3, 0)[0])))

##check
# GMM = GMM(n_components = 3, n_init = 10)
# GMM.fit(X)
# print(GMM.predict_proba(X))
