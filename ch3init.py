from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import  matplotlib.pyplot as plt

import numpy as np

from funcs import plotDecisionRegions

iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(xTrain)

xTrainStd = sc.transform(xTrain)
xTestStd = sc.transform(xTest)

#ppn = Perceptron(eta0=0.1, random_state=1)
lr = LogisticRegression(C=10.0, random_state=21, solver='lbfgs', multi_class='multinomial')
lr.fit(xTrainStd, yTrain)

yPred = lr.predict(xTestStd)
print("Missclassified examples: %d" % (yTest != yPred).sum())
print("Accuracy: %.3f" % accuracy_score(yTest, yPred))

xCombinedStd = np.vstack((xTrainStd, xTestStd))
yCombined = np.hstack((yTrain, yTest))

plotDecisionRegions(X=xCombinedStd, y=yCombined, classifier=lr, testIdx=range(105, 150))
plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

