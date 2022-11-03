import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("survey lung cancer.csv")

for i in df.columns[2:-1]:
  df[i].replace(1,0,inplace =True)
  df[i].replace(2,1,inplace =True)
df.head()

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'LUNG_CANCER'.
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])

from sklearn.model_selection import train_test_split

X= df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']

X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestClassifier

# create regressor object
regressor = RandomForestClassifier(n_estimators=1000, random_state=0)

# fit the regressor with x and y data
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn import metrics

print(y_pred)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# # save the classifier
# with open('Models/random_forest_classifier.pkl', 'wb') as fid:
#     pickle.dump(regressor, fid)

# load it again
with open('Models/random_forest_classifier.pkl', 'rb') as fid:
    model_loaded = pickle.load(fid)

print(model_loaded.predict(X_test))
print(X_test.shape)