
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

x = np.array([['Texas'], ['California'], ['Texas'], ['Delaware'], ['Texas']])

one_hot = LabelBinarizer()

print(one_hot.fit_transform(x))
print(one_hot.classes_)

print(np.array(['California', 'Delaware', 'Texas'], dtype='<U10'))

print(pd.get_dummies(x[:,0]))







