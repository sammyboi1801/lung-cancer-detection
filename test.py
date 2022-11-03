import matplotlib.pyplot as plt
import cv2
from skimage.io import imread_collection
import numpy as np
import tensorflow as tf


#PREPARING TRAINING DATA
train_dir1 = 'Datasets/Data/train/adenocarcinoma/*.png'
train_dir2 = 'Datasets/Data/train/large.cell.carcinoma/*.png'
train_dir3 = 'Datasets/Data/train/squamous.cell.carcinoma/*.png'
train_dir4 = 'Datasets/Data/train/normal/*.png'

#creating a collection with the available images
train1 = imread_collection(train_dir1)
train1=np.array(train1,dtype=object)
train2 = imread_collection(train_dir2)
train2=np.array(train2,dtype=object)
train3 = imread_collection(train_dir3)
train3=np.array(train3,dtype=object)
train4 = imread_collection(train_dir4)
train4=np.array(train4,dtype=object)

print(train1.shape)
print(train2.shape)
print(train3.shape)
print(train4.shape)

X_train=np.append(train1,train2)
X_train=np.append(X_train,train3)
X_train=np.append(X_train,train4)

for i in range(len(X_train)):
    X_train[i]=cv2.cvtColor(X_train[i],cv2.COLOR_RGBA2RGB)  #converts rgba to rgb
    X_train[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)   #converts rgb to grayscale
    X_train[i] = X_train[i] / 255

Y_train=np.ones(train1.shape[0]+train2.shape[0]+train3.shape[0])
temp=np.zeros(train4.shape[0])
Y_train=np.append(Y_train,temp)


temp=np.expand_dims(cv2.resize(X_train[0],dsize=(500,400),interpolation=cv2.INTER_CUBIC),axis=0)
print(temp.shape)

print("Preparing",end="")
for i in range(1,len(X_train)):
    temp=np.append(temp,np.expand_dims(cv2.resize(X_train[i], dsize=(500, 400), interpolation=cv2.INTER_CUBIC),axis=0),axis=0)

X_train=temp


model = tf.keras.models.load_model('Models/model2')

loss, acc = model.evaluate(X_train, Y_train, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# model.summary()