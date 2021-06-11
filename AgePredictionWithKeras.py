# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import cv2
import matplotlib.image as mpimg
import sklearn
import random

imgPath = [] # This is an empyt list for images's paths
# %%
path = "/home/kamber/Desktop/KamberAgcan/Python/Age Prediction/20-50/train"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        imgPath.append(os.path.join(dirname, filename)) # The paths read from the file are thrown into the created empty list.
imgPath = np.asarray(imgPath)  # Converted to list array format containing paths.
print(imgPath.shape,type(imgPath))

imgShape = imgPath.shape[0] # The shape of the imgPath array is assigned to a variable named imgShape in order to use it in loops.
print("imgShape: ",imgShape,type(imgShape))

random.shuffle(imgPath)    # Since the size of the images in the data set is large, 5000 images were randomly selected, 
                           # and training will be made with these selected images.

whatkindofimage = []        # Empyt list                 
for i in range(0,imgShape):
    i = path.split("/")[9]       # We dynamically determine whether the test image or the train image.
    whatkindofimage.append(i)
print(whatkindofimage[0])

imgName = []           # Empyt list 
for k in range(0,imgShape):   
    k = imgPath[k].split("/")[11]   # The names of the images were determined from the path using the split method.
    imgName.append(k) 
print(imgName[0])   

Age =[]
for i in range(0,imgShape):   
    i = i + 1
    k = imgPath[i-1].split("/")[10] # The ages of the images were determined.
    Age.append(k)
print(Age[0])

# Create data frame
c1 = pd.Series(whatkindofimage)
c2 = pd.Series(imgName)
c3 = pd.Series(Age)
c4 = pd.Series(imgPath)
data_ = dict(whatkindofimage = c1,imgName = c2,Age = c3, imgPath = c4) # The created columns are added to the dataframe.
df = pd.DataFrame(data_)
print(df.head())

df["imgPath"][0]

def imreading(index):    
    x = imgPath[index]   #  To read the images with the imread method, a method has been developed to get the index of 
    y = plt.imread(x)    #  the path in the name of imread, and read the photo in the desired index in array format.
    return y
imreading(0)

def imshowing(index):                # Developed a method called imshowing to read images.
    x = imgPath[index]               # With determined index, the photo in the desired order in the path can be visualized.
    y = ("Age =",loopsAndframe(imgShape,imgPath).Age[index]) # the age in the photograph was written to x axis.
    z = (loopsAndframe(imgShape,imgPath).whatkindofimage[0],loopsAndframe(imgShape,imgPath).imgName[index])               
    k = mpimg.imread(x) 
    plt.imshow(k)  
    plt.title(z)                  # The name of the visualized photo is printed in the title section.
    #plt.axis("off")
    plt.xlabel(y)
    plt.show()
imshowing(0)

model_img1 = []
model_img = []
for i in range(0,300):
    k = (df["imgPath"][i])
    model_img1.append(k)
    l = cv2.imread(model_img1[i])
    model_img.append(l)

model_img = np.asarray(model_img)
print(model_img.shape)

X = np.asarray(model_img[0:2000])
print(X.shape)

X = X.reshape(-1,128,128,3).astype("float")
print(X.shape,type(X))

model_age = []
for i in range(0,300):
    k = (df["Age"][i])
    l = cv2.imread(model_img1[i])
    model_age.append(k)

model_img = np.asarray(model_img)
print(model_img.shape)

X = np.asarray(model_img)
print(X.shape,type(X))

X = X.reshape(-1,128,128,3).astype("float") # setting images entries for model
print(X.shape,type(X))

# visualize number of digits classes
plt.figure(figsize=(15,7))
sns.countplot(model_age, palette="icefire")
plt.title("Number of digit classes")
plt.show()

Y = np.asarray(model_age).reshape(-1,1) #setting label entries for model
print(Y.shape,type(Y))

# normalization
X = X/255.0
Y = Y/255.0
print(X.shape,type(X))
print(Y.shape,type(Y))

plt.imshow(X[0])           #view of the same image after normalization
plt.show()

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y = to_categorical(Y, num_classes = 51)
print(Y.shape)

# %% Created Model
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,ZeroPadding2D,Activation
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = "Same", activation = "relu", input_shape = (128,128,3)))
model.add(Conv2D(140,(3,3),activation="relu"))
model.add(Conv2D(130,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Conv2D(120,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(51))
model.add(Activation("softmax"))

optimizer = Adam(lr = 0.001,beta_1 = 0.9, beta_2 = 0.999)

# Compile the Model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
                                   # loss = "categorical_crossentropy" - "binary_crossentropy"  - "sparse_categorical_crossentropy"

# Epochs and Batch Size
epochs = 15 # for better result increase the epochs
batch_size = 100
    
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# fit the model
history = model.fit(datagen.flow(X_train,Y_train,batch_size = batch_size), epochs = epochs, 
                              validation_data = (X_val,Y_val), steps_per_epoch = X_train.shape[0]// batch_size)

# %% visualize results
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="Validation Loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['accuracy'], color='r', label="Accuracy")
#plt.plot(history.history["val_accuracy,"],color = "r",label = "val_accuracy")
plt.title("Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% Saving model
model.save("Age_load_model")











