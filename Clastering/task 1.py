import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import random
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

sns.set()
f, (ax1, ax2) = plt.subplots(1, 2)
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
ax1.scatter(X[:, 0], X[:, 1], s=20)
# Незашумленные данные
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2)
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
# Зашумленные данные
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.show()


f, ax1 = plt.subplots(1, 1)
iris = load_iris()
pca = PCA(n_components=2).fit(iris.data)
pca_2d = pca.transform(iris.data)
# Данные из задания 2
kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_2d)
y_kmeans = kmeans.predict(pca_2d)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
ax1.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
plt.show()