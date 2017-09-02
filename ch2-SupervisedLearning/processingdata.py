"""

    Processing the data


"""
import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1,-2.9,3.3],[-1.2,7.8,-6.1],[3.9,0.4,2.1],[7.3,-9.9,-4.5]])

"""Binarization Data"""


"""
    This process is used when we want to convert our numerical values into boolean values.
"""

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print('\nBinarized data:\n', data_binarized)


""" Mean removal"""


"""
    Removing the mean is a common preprocessing technique used in machine learning.
    It`s useful to remove the mean from our feature vector, so that each feature is centered on zero.
    We do this in order to remove bias from the features in our feature vector.
"""
#print mean and standard deviation
print("\nBEFORE: ")
print("Mean =",input_data.mean(axis=0))
print("Std deviation =",input_data.std(axis=0))

#Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =",data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))



""" Scaling """


"""
    In our feature vector, the value of each feature can vary between many random values.
    So it becomes important to scale those features so that it isa a level playing field for machine learning algorithm to train on. 

"""

#Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin Max scaled data:\n",data_scaled_minmax)
