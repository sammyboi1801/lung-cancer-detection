from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

tensor=-99
rfc=-99


def delete_files():
    files = glob.glob('D:\pythonProject\LungCancerCTScan\LungCancerDetectionApp\LungCancerDetectionApp\media\*')
    files1 = glob.glob('D:\pythonProject\LungCancerCTScan\LungCancerDetectionApp\LungCancerDetectionApp\static\output_img\*')
    for f in files:
        os.remove(f)

    for f in files1:
        os.remove(f)

def move_files():
    import os
    import shutil

    source = 'D:\pythonProject\LungCancerCTScan\LungCancerDetectionApp\LungCancerDetectionApp\media'
    destination = 'D:\pythonProject\LungCancerCTScan\LungCancerDetectionApp\LungCancerDetectionApp\static\output_img'

    # gather all files
    allfiles = os.listdir(source)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)

def tensorflow_model():
    model = tf.keras.models.load_model('D:\pythonProject\LungCancerCTScan\Models\model2')
    file_dir='D:/pythonProject/LungCancerCTScan/LungCancerDetectionApp/LungCancerDetectionApp/media/image.png'

    image=cv2.imread(file_dir)
    image = np.array(image)

    image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image=image / 255

    image=np.expand_dims(cv2.resize(image,dsize=(500,400),interpolation=cv2.INTER_CUBIC),axis=0)

    print(image.shape)
    predict=model.predict(image)
    tensor = predict
    if(predict[0]<0.5):
        print('No cancer!!')
    else:
        print("There's a chance you have cancer")
    print('Image Prediction: ',predict)
    return tensor


def RFC_mod( gender, age, smoke, yellow_fingers, anxiety, peer_pressure, chronic, fatigue, allergies, wheezing,alcohol,cough,breath,swallow,chest):
    outputs = ['You have a very chance of having Lung Cancer according to our diagnosis. However, this cannot be a substitute for real diagnosis. Please check with a professional.',
               'You have a moderate chance of having Lung Cancer. We would recommend you to']

    if (gender == 'Male'):
        X = [1, age, smoke, yellow_fingers, anxiety, peer_pressure, chronic, fatigue, allergies, wheezing,alcohol,cough,breath,swallow,chest]
    else:
        X=[0, age, smoke, yellow_fingers, anxiety, peer_pressure, chronic, fatigue, allergies, wheezing,alcohol,cough,breath,swallow,chest]





    # print(X)

    with open('D:/pythonProject/LungCancerCTScan/Models/random_forest_classifier.pkl', 'rb') as fid:
        model_loaded = pickle.load(fid)

    try:
        rfc=model_loaded.predict([X])
        return rfc
    except:
        # print('Exception')
        pass

def home(request):
    # fetch current session number
    delete_files()
    smoke=-1
    yellow_fingers=-1
    anxiety=-1
    peer_pressure=-1
    chronic=-1
    fatigue=-1
    allergies=-1
    wheezing=-1
    alcohol=-1
    cough=-1
    breath=-1
    swallow=-1
    chest=-1
    age=-1
    gender=-1
    # files=None

    if(request.method=='POST'):
        # delete_files()
        smoke=request.POST['smoke']
        yellow_fingers=request.POST['yellow-fingers']
        anxiety = request.POST['anxiety']
        peer_pressure = request.POST['peer-pressure']
        chronic = request.POST['chronic']
        fatigue = request.POST['fatigue']
        allergies = request.POST['allergies']
        wheezing = request.POST['wheezing']
        alcohol = request.POST['alcohol']
        cough = request.POST['cough']
        breath = request.POST['breath']
        swallow = request.POST['swallow']
        chest = request.POST['chest']
        age = request.POST['age']
        gender = request.POST['gender']
        files = request.FILES['CT_scan']
        fs=FileSystemStorage()
        fs.save(name='image.png',content=files)
        print('image',files.size)
        tensor=tensorflow_model()
        # request.session['tensor']={'tensor':tensor[0]}
        # request.session['tensor'] = tensor[0]
        if (smoke == 'option1'):
            smoke = 1
        else:
            smoke = 0

        if (yellow_fingers == 'option3'):
            yellow_fingers = 1
        else:
            yellow_fingers = 0

        if (anxiety == 'option5'):
            anxiety = 1
        else:
            anxiety = 0

        if (peer_pressure == 'option7'):
            peer_pressure = 1
        else:
            peer_pressure = 0

        if (chronic == 'option9'):
            chronic = 1
        else:
            chronic = 0

        if (fatigue == 'option11'):
            fatigue = 1
        else:
            fatigue = 0

        if (allergies == 'option13'):
            allergies = 1
        else:
            allergies = 0

        if (wheezing == 'option15'):
            wheezing = 1
        else:
            wheezing = 0

        if (alcohol == 'option17'):
            alcohol = 1
        else:
            alcohol = 0

        if (cough == 'option19'):
            cough = 1
        else:
            cough = 0

        if (breath == 'option21'):
            breath = 1
        else:
            breath = 0

        if (swallow == 'option23'):
            swallow = 1
        else:
            swallow = 0

        if (chest == 'option25'):
            chest = 1
        else:
            chest = 0


        print(smoke, yellow_fingers, anxiety, peer_pressure, chronic, fatigue, allergies, wheezing, age, gender)
        rfc = RFC_mod(gender, age, smoke, yellow_fingers, anxiety, peer_pressure, chronic, fatigue, allergies, wheezing,
                      alcohol, cough, breath, swallow, chest)

        # request.session['rfc_pred']={'rfc':rfc[0]}
        # request.session['rfc_pred'] = rfc[0]
        # print('Image: ',tensor)
        print('Survey Prediction: ',rfc)

    if(age!=-1 and gender!=-1):
        move_files()
        return redirect('/output')
    return render(request,'home.html')


def output(request):
    # images_location = glob.glob(
    #     'D:\pythonProject\LungCancerCTScan\LungCancerDetectionApp\LungCancerDetectionApp\media\*')
    # print('hmmm: ', images_location)
    outputs = [
        'You have a high chance of having Lung Cancer according to our diagnosis. However, this cannot be a substitute for real diagnosis. Please check with a professional.',
        'You have a moderately high chance of having Lung Cancer. We would recommend you to get a check-up',
        'You have a moderate chance of having Lung Cancer. We would recommend you to get a check-up',
        'You have a low chance having Lung Cancer. We would still suggest you to get a check up.']
    final=''
    # tensor=request.session['tensor']
    # rfc=request.session['rfc_pred']

    if(tensor>0.5 and rfc!=0):
        final=outputs[0]
    elif(tensor>0.5 and rfc!=0):
        final=outputs[1]
    elif (tensor < 0.5 and rfc != 0):
        final = outputs[2]
    else:
        final = outputs[3]
    # tensor=request.session['tensor']
    # rfc= request.session['rfc_pred']
    # print(rfc)
    return render(request, 'output.html',{'statement':final})