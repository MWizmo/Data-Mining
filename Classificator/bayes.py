from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

iris = load_iris()
recall, precision, f1_score, accuracy = [], [], [], []
test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['First class', 'Second class', 'Third class'])
    rows = report.split('\n')
    prec, rec, f1 = (sum([float(rows[i].split()[j]) for i in [2, 3, 4]]) / 3.0 for j in [2, 3, 4])
    acc = float(rows[6].split()[1])
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)
f, axes = plt.subplots(2, 2)
axes[0, 0].plot(test_sizes, accuracy, c='green')
axes[0, 0].set_xlabel('Test size')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 1].plot(test_sizes, f1_score, c='purple')
axes[0, 1].set_xlabel('Test size')
axes[0, 1].set_ylabel('F1 score')
axes[1, 0].plot(test_sizes, recall, c='orange')
axes[1, 0].set_xlabel('Test size')
axes[1, 0].set_ylabel('Recall')
axes[1, 1].plot(test_sizes, precision, c='red')
axes[1, 1].set_xlabel('Test size')
axes[1, 1].set_ylabel('Precision')
plt.show()
