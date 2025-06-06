import numpy as np
import kmeans
import common
import naive_em
import em
import time
import matplotlib.pyplot as plt

X = np.loadtxt("toy_data.txt")
Xx = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")
x_netflix = np.loadtxt("netflix_incomplete.txt")
x_netflix_gold = np.loadtxt("netflix_complete.txt")


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

# KMeans_plot()

###TASK 3

print(naive_em.estep(X, common.init(X, 2, 0)[0]))
# print(naive_em.mstep(X, naive_em.estep(X, common.init(X, 3, 0)[0])[0]))
# print(naive_em.run(X, common.init(X, 3, 0)[0], naive_em.estep(X, common.init(X, 3, 0)[0])))

print(naive_em.run(X, common.init(X, 2, 0)[0], common.init(X, 3, 3)[1]))


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
            common.plot(X, mixture, post, f"GMM, K = {k}")

# GMM_plot()

##TASK 5

def run_BIC():
    l1 = []
    for s in range(0, 5):
        l = []
        for k in range(1, 5):
            mixture, post = common.init(X, k, s)
            mixture, post, LL = naive_em.run(X, mixture, post)
            l.append(common.bic(X, mixture, LL))
        l1.append(l)
    return l1

# print(run_BIC())

#TASK 8

def LL_max(X):
    l1 = []
    for s in range(0, 5):
        l = []
        for k in (1, 12):
            print(k)
            mixture, post = common.init(X, k, s)
            l.append(em.run(X, mixture, post)[2])
        l1.append(l)
    return l1


# start = time.time()
# print(LL_max(x_netflix))
# end = time.time()
# print(f"this computation took {round((end - start)/60)} minutes")

#COMPLETING MISSING ENTRIES
#the mixture here should be a final one
def missing_entries(X, mixture):

    return(em.fill_matrix(X, mixture))


# mixture1, post1 = common.init(x_netflix, 12, 1)
# mixture2 = em.run(x_netflix, mixture1, post1)[0]
# X_new = missing_entries(x_netflix, mixture2)
# print(common.rmse(X_new, x_netflix_gold))

# def missing_entries1(X, mixture):
#
#     return(em.fill_matrix(X, mixture))
#
#
# mixture1, post1 = common.init(Xx, 3, 0)
# mixture2 = em.run(Xx, mixture1, post1)[0]
# print(missing_entries1(Xx, mixture2))