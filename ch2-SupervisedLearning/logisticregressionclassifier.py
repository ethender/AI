"""
    Logistic Regression classifier:

            Logistic regression is a technique that is used to explain the relationship between
            input variables and output variables. The input variables are assumed to be indpendent and
            the output variable is referred to as the dependent variable.

            The dependent variable can take only a fixed set of values.
            These values correspond to the classes of the clasification problem.

"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


from utilities import visualize_classifier



X  = np.array([
    [3.1,7.2],[4,6.7],
    [2.9,8],[5.1,4.5],[6,5],[5.6,5],
    [3.3,0.4],[3.9,0.9],[2.8,1],[0.5, 3.4],
    [1,4],[0.6,4.9]
    ])

y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])


classifier = linear_model.LogisticRegression(solver='liblinear',C=1)
classifier.fit(X,y)

visualize_classifier(classifier,X,y)

