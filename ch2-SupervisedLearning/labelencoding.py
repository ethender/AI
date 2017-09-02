"""
    Label Encoding:

        When we perform classification, we usaully deal with a lot of labels.
        These labels can be in the form of words, numbers, or something else.
        The machine learning functions in sklearn expect them to be numbers.

"""


import numpy as np
from sklearn import preprocessing

#sample labels
input_labels = ['red','black','red','green','black','yellow','white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#prints the label encoding
print("\nLabel mapping:")
for i,item in enumerate(encoder.classes_):
    print(item,'-->',i)
    
#tells label encoding value
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print('\nLabels =',test_labels)
print('Encoded values=',list(encoded_values))


encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print('\nEncoded values=',encoded_values)
print('\nDecoded labels =',list(decoded_list))
