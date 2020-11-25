from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import random


iris = load_iris()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(iris.data)
# Незашумленные данные
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

plt.legend([c1, c2, c3], ['Кластер 1', 'Кластер 2', 'Шум'])
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
a = 0
for point in X:
    num = random.randint(1, 20)
    if num == 5 or num == 6:
        point[1] += random.uniform(-3.5, 3.5)
        a += 1
    elif num == 15 or num == 16:
        point[0] += random.uniform(-4.5, 4.5)
        a += 1
ax1.scatter(X[:, 0], X[:, 1], s=20)
print(f'Изменено {a}/{len(X)} точек')
# Зашумленные данные из задания 1
dbscan.fit(X)
for i in range(0, X.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = ax2.scatter(X[i, 0], X[i, 1], c='r', marker='o')
    elif dbscan.labels_[i] == 1:
        c2 = ax2.scatter(X[i, 0], X[i, 1], c='g', marker='o')
    elif dbscan.labels_[i] == 2:
        c3 = ax2.scatter(X[i, 0], X[i, 1], c='m', marker='o')
    elif dbscan.labels_[i] == 3:
        c4 = ax2.scatter(X[i, 0], X[i, 1], c='y', marker='o')
    elif dbscan.labels_[i] == -1:
        c5 = ax2.scatter(X[i, 0], X[i, 1], c='b', marker='*')

ax2.legend([c1, c2, c3, c4, c5], ['Кластер 1', 'Кластер 2', 'Кластер 3', 'Кластер 4', 'Шум'])
plt.show()