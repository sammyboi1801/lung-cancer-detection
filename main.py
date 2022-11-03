import matplotlib.pyplot as plt
import cv2
from skimage.io import imread_collection
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, ReLU, Conv2D, MaxPooling2D, Dense,Flatten
import cv2
from sklearn.model_selection import train_test_split


'''-------------------------------------------------------------------------------------------------------'''



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


X_train=np.append(train1,train2)
X_train=np.append(X_train,train3)
X_train=np.append(X_train,train4)

print("TRAINING")
#printing the number of images in different folders
print("Cancer: ",train1.shape)
print("Cancer: ",train2.shape)
print("Cancer: ",train3.shape)
print("Normal: ",train4.shape)


for i in range(len(X_train)):
    X_train[i]=cv2.cvtColor(X_train[i],cv2.COLOR_RGBA2RGB)  #converts rgba to rgb
    X_train[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)   #converts rgb to grayscale
    # X_train[i] = X_train[i] / 255

Y_train=np.ones(195+115+155)
temp=np.zeros(136)
Y_train=np.append(Y_train,temp)


temp=np.expand_dims(cv2.resize(X_train[0],dsize=(500,400),interpolation=cv2.INTER_CUBIC),axis=0)
# print(temp.shape)

print("Preparing",end="")
for i in range(1,len(X_train)):
    temp=np.append(temp,np.expand_dims(cv2.resize(X_train[i], dsize=(500, 400), interpolation=cv2.INTER_CUBIC),axis=0),axis=0)
    # if(i==(int(len(X_train)/3))):
    #     print('.',end="")
    # elif(i==(int(len(X_train)/1.3))):


    #     print('.', end="")
    # elif (i == (int(len(X_train)/8))):
    #     print('.', end="")
    # print(temp.shape)


X_train=temp
temp=cv2.Canny(X_train[0],60,120)
pro=np.expand_dims(cv2.addWeighted(X_train[i],0.7,temp,0.4, 0),axis=0)
for i in range(1,len(X_train)):
    # temp=np.append(temp,cv2.Canny(X_train[i],100,200))
    # temp=np.append(temp, np.expand_dims(cv2.Canny(X_train[i],60,120), axis=0), axis=0)
    temp=cv2.Canny(X_train[i],60,120)
    pro = np.append(pro, np.expand_dims(cv2.addWeighted(X_train[i],0.7,temp,0.4, 0),axis=0), axis=0)
    # cv2.addWeighted(X_train[i], 0.7, temp, 0.4, 0)

pro=pro/255
print(np.max(X_train[1]),pro.shape)

# for i in range(len(X_train)):
#     X_train[i]=X_train[i].astype('float32')/255.0


'''-------------------------------------------------------------------------------------------------------'''


# #PREPARING TESTING DATA
# test_dir1 = 'Datasets/Data/test/adenocarcinoma/*.png'
# test_dir2 = 'Datasets/Data/test/large.cell.carcinoma/*.png'
# test_dir3 = 'Datasets/Data/test/squamous.cell.carcinoma/*.png'
# test_dir4 = 'Datasets/Data/test/normal/*.png'
#
#
# #creating a collection with the available images
# test1 = imread_collection(test_dir1)
# test1=np.array(test1,dtype=object)
# test2 = imread_collection(test_dir2)
# test2=np.array(test2,dtype=object)
# test3 = imread_collection(test_dir3)
# test3=np.array(test3,dtype=object)
# test4 = imread_collection(test_dir4)
# test4=np.array(test4,dtype=object)
#
# #printing the number of images in test folders
# print()
# print("TESTING")
# print("Cancer: ",test1.shape)
# print("Cancer: ",test2.shape)
# print("Cancer: ",test3.shape)
# print("Normal: ",test4.shape)
#
# X_test=np.append(test1,test2)
# X_test=np.append(X_test,test3)
# X_test=np.append(X_test,test4)
#
#
# for i in range(len(X_test)):
#     X_test[i]=cv2.cvtColor(X_test[i],cv2.COLOR_RGBA2RGB)  #converts rgba to rgb
#     X_test[i] = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY)   #converts rgb to grayscale
#     X_test[i] = X_test[i] / 255
#
# Y_test=np.ones(120+51+90)
# temp=np.zeros(54)
# Y_test=np.append(Y_test,temp)
#
# temp=np.expand_dims(cv2.resize(X_test[0],dsize=(500,400),interpolation=cv2.INTER_CUBIC),axis=0)
#
# print("Preparing",end="")
# for i in range(1,len(X_test)):
#     temp=np.append(temp,np.expand_dims(cv2.resize(X_test[i], dsize=(500, 400), interpolation=cv2.INTER_CUBIC),axis=0),axis=0)
#     # if (i == (int(len(X_test)/3))):
#     #     print('.', end="")
#     # elif (i == (int(len(X_test)/1.3))):
#     #     print('.', end="")
#     # elif (i == (int(len(X_test)/8))):
#     #     print('.', end="")
#     # print(temp.shape)
#
# X_test=temp
#
# X=np.append(X_train,X_test,axis=0)
# Y=np.append(Y_train,Y_test,axis=0)
# print(X.shape,Y.shape)
# dataset=np.array([X_train,X_test,Y_train,Y_test],dtype='object')
# print(dataset.shape)
# x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
#
#
#
# '''-------------------------------------------------------------------------------------------------------'''
# print()
# print()
# print("X train: ",x_train.shape)
# print("Y train: ",y_train.shape)
# print("X test: ",x_test.shape)
# print("Y test: ",y_test.shape)
'''-------------------------------------------------------------------------------------------------------'''

#
# model=Sequential()
#
# model.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',input_shape=(400,500,1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Conv2D(50,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
#
# model.add(Conv2D(50,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
#
# model.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
#
# # model.add(Conv2D(50,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))
# # model.add(MaxPooling2D(pool_size=(3,3)))
#
# model.add(Flatten())
# # model.add(Dense(500,activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(Dense(250,activation='relu'))
# # model.add(Dropout(0.4))
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.7))
# model.add(Dense(50,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(25,activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1,activation='sigmoid'))
#
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
# model.summary()
# print()
# # _=input("Press any key: ")
#
# model.fit(x_train,y_train, batch_size=8,epochs=10,validation_data=(x_test,y_test))
#
# model.save('Models/model2')







plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(pro[0],cmap='gray')
plt.title('hmm')
# plt.subplot(1,2,2)
# plt.imshow(X_train[200],cmap='gray')
# plt.title('X_train')
# plt.subplot(1,5,3)
# plt.imshow(x_train[150],cmap='gray')
# plt.title(y_train[150])
# plt.subplot(1,5,4)
# plt.imshow(x_test[60],cmap='gray')
# plt.title(y_test[60])
# plt.subplot(1,5,5)
# plt.imshow(x_test[26],cmap='gray')
# plt.title(y_test[26])
plt.show()