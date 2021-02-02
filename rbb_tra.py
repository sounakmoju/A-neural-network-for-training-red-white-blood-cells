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
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn.preprocessing import LabelBinarizer
INIT_LR = 0.0007
EPOCHS= 100
BS = 64
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
path_1=r'C:\\reasearch\rbb\train'
path_2=r'C:\\reasearch\rbb\valid'
os.chdir(r'C:\\reasearch\rbb\train')

filenames = glob.glob("**/*.jpg",recursive = True)
filenames.sort()
image_1=[]

csv_ind=df.loc[:,'filename']

for img in (filenames):
    images=cv2.imread(img)
    
   
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
                

os.chdir(path_2)
filenames1 = glob.glob("**/*.jpg",recursive = True)
filenames1.sort()

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
x_train= np.array(train_images)
y_train= np.array(train_classes)
x_val=np.array(vali_images)
y_val=np.array(vali_classes)
dropout1=Dropout(0.5)
dropout2=Dropout(0.5)
dropout3=Dropout(0.5)
model_vgg16_conv=VGG16(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()
input=Input(shape=(224,224,3),name='image_input')
output_vgg16_conv=model_vgg16_conv(input)
x=Flatten(name='flatten')(output_vgg16_conv)
#x=Flatten(name='flatten')(output_vgg16_conv)
x=Dense(2048,activation='relu',name='fc1')(x)
x=dropout1(x)
#x=(Dropout(0.5))
##my_model.add(Dropout(0.5))
x=Dense(1024,activation='relu',name='fc2')(x)
x=dropout2(x)
#my_model.add_loss(Dropout(0.5))
#x=Dense(1024,activation='relu',name='fc3')(x)
x=Dense(512,activation='relu',name='fc3')(x)
x=dropout3(x)
x=Dense(3,activation='softmax',name='predictions')(x)
my_model=Model(inputs=input,outputs=x)
opt = Adam(INIT_LR)
my_model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
#model_final.summary()

my_model.summary()
lb = LabelBinarizer()
y_tr =lb.fit_transform(y_train)
y_vl=lb.fit_transform(y_val)
trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=x_train, y=y_tr)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=x_val, y=y_vl)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
hist = my_model.fit_generator(generator= trdata.flow(x=x_train,y=y_tr,batch_size=BS), steps_per_epoch= (len(x_train) // BS), epochs=EPOCHS, validation_data= testdata,validation_steps=2,callbacks=[checkpoint,early])




