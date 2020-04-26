
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

x = np.array([['Texas'], ['California'], ['Texas'], ['Delaware'], ['Texas']])

one_hot = LabelBinarizer()

print(one_hot.fit_transform(x))
print(one_hot.classes_)

print(np.array(['California', 'Delaware', 'Texas'], dtype='<U10'))

print(pd.get_dummies(x[:,0]))


from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()

enc.fit([[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]]
        )

array =enc.transform([[0,1,3]]).toarray()

print(array)




