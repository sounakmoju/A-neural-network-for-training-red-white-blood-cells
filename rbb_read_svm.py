##import numpy as np 
import pandas as pd 
import os
from pathlib import Path
from int_rect import get_iou
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.layers import Input,Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
train_images=[]
train_classes=[]
vali_images=[]
vali_classes=[]
df = pd.read_csv('_annotations_train.csv')
df1=pd.read_csv('_annotations_valid.csv')
def transform_label(value):
    if value=='RBC':
        return 0
    elif value=='RBC':
        return 1
    else:
        return 2
df["class_1"]=df.class_1.apply(transform_label)
df1["class_1"]=df1.class_1.apply(transform_label)
path_1=r'C:\\Downloads\reasearch\rbb\train'
path_2=r'C:\\Downloads\reasearch\rbb\valid'
os.chdir(r'C:\\Downloads\reasearch\rbb\train')
#os.chdir(path_1)
filenames = glob.glob("**/*.jpg",recursive = True)
filenames.sort()
image_1=[]
#print(int(df.loc[0,'xmin']))
csv_ind=df.loc[:,'filename']
#print((csv_ind[2777]))

#print(filenames)
for img in (filenames):
    images=cv2.imread(img)
    #filename = img.split(".")[0]+".jpg"
#print(str(img))
   
    idx = np.where((csv_ind)==(img))
    imout = images.copy()
    if idx[0].shape[0]:
        for i in idx:
            for j in i:
                x1 = int(df.loc[j,'xmin'])
                y1 = int(df.loc[j,'ymin'])
                x2 = int(df.loc[j,'xmax'])
                y2 = int(df.loc[j,'ymax'])
                w=(x2-x1)
                h=(y2-y1)
                timage = imout[y1:y1+h,x1:x1+w]
                #print(timage.shape)
                try:
                    resized = cv2.resize(timage,(224,224),interpolation = cv2.INTER_AREA)
                except Exception as e:
                    print(str(e))
                train_images.append(resized)
                train_classes.append(df.loc[j,"class_1"])
                #print(resized.shape)
                #print(train_classes)
                
                #print(w,h)
                #cv2.rectangle(images, (x1, y1), (x2, y2), (255,0,0), 2)
                #plt.imshow(resized)
                #plt.show()
                #cv2.waitKey(0)
            
    
    
    #x1 = int(df.loc[e,'xmin'])
    #y1 = int(df.loc[e,'ymin'])
    ##x2 = int(df.loc[e,'xmax'])
    #y2 = int(df.loc[e,'ymax'])
    #cv2.rectangle(images, (x1, y1), (x2, y2), (255,0,0), 2)
    #plt.imshow(images)
    #plt.show()
#images = [cv2.imread(img) for img in filenames]
#print(len(images))

os.chdir(path_2)
filenames1 = glob.glob("**/*.jpg",recursive = True)
filenames1.sort()
#image_1=[]
#print(int(df.loc[0,'xmin']))
csv_ind1=df1.loc[:,'filename']
#print((csv_ind[2777]))

#print(filenames)
for img1 in (filenames1):
    images1=cv2.imread(img1)
    #filename = img.split(".")[0]+".jpg"
#print(str(img))
   
    idx_1 = np.where((csv_ind1)==(img1))
    imout1 = images1.copy()
    if idx_1[0].shape[0]:
        for i in idx_1:
            for j in i:
                x_1 = int(df.loc[j,'xmin'])
                y_1 = int(df.loc[j,'ymin'])
                x_2 = int(df.loc[j,'xmax'])
                y_2 = int(df.loc[j,'ymax'])
                w=(x_2-x_1)
                h=(y_2-y_1)
                timage1 = imout1[y_1:y_1+h,x_1:x_1+w]
                try:
                    resized1 = cv2.resize(timage1,(224,224),interpolation = cv2.INTER_AREA)
                except Exception as e:
                    print(str(e))
                vali_images.append(resized1)
                vali_classes.append(df1.loc[j,"class_1"])
for i in train_images:
    #gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    plt.imshow(i)
    plt.show()
    

##for i in range(len(train_images)):
####    print(train_images[i].shape)
##for j in range(len(vali_images)):
##print(vali_images[j].shape)
#print(len(train_images))
#print(len(train_classes))
#print(len(vali_images))
#print(len(vali_classes))





