
from matplotlib.colors import ListedColormap
import  matplotlib.pyplot as plt

import numpy as np

def plotDecisionRegions(X, y, classifier, testIdx=0, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1Min, x1Max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2Min, x2Max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1Min, x1Max, resolution), np.arange(x2Min, x2Max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    if testIdx:
        xTest, yTest = X[testIdx, :], y[testIdx]

        plt.scatter(xTest[:, 0], xTest[:, 1], c='none', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label="Test Set")