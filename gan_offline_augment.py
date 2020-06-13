import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py
from keras.utils.io_utils import HDF5Matrix
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Activation,Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.optimizers import SGD,Adam
from time import time
from sklearn.utils import shuffle
from glob import glob
from tqdm import tqdm
import pickle
from skimage.io import imsave
from sklearn.utils.class_weight import compute_class_weight

HM_EPOCHS = 30
batch_size = 50

IMG_SIZE = 64

from numpy.random import seed
seed(101)

from tensorflow import set_random_seed
set_random_seed(101)


#optimizer=Adam(lr=0.001)
def find_last_checkpoint(disease):
    last_model_point =0
    for f in (glob('saved_model/generator_{}_*'.format(disease))):
        file = f.split('/')[-1]
        checkpoint_no = int(file.split('_')[-1])
        if checkpoint_no > last_model_point:
            last_model_point = checkpoint_no

    print ("Loading generator model %d for disease %s" %(last_model_point,disease))
    return int(last_model_point)

labels = ['nv','bcc','bkl','mel']
index_to_labels = {3:'bcc', 0:'nv', 2:'bkl', 1:'mel'}
labels_to_index = {'nv':0, 'mel':1,'bkl':2,'bcc':3}
disease_counts={0:500,1:1000,2:1000,3:1500}

generator_model={}
for disease in labels:
    generator_model[disease]=load_model('saved_model/generator_{}_{}'.format(disease,find_last_checkpoint(disease)))



f=open('train_x.pickle','rb')
train_x=pickle.load(f)
f.close()

f=open('train_y.pickle','rb')
train_y=pickle.load(f)
f.close()

f=open('validation_x.pickle','rb')
validation_x=pickle.load(f)
f.close()

f=open('validation_y.pickle','rb')
validation_y=pickle.load(f)
f.close()

f=open('test_x.pickle','rb')
test_x=pickle.load(f)
f.close()

f=open('test_y.pickle','rb')
test_y=pickle.load(f)
f.close()

print (train_x.shape,train_y.shape,validation_x.shape,validation_y.shape,test_x.shape,test_y.shape)
train_x = train_x/255.0
validation_x = validation_x/255.0
test_x = test_x/255.0

print (np.amin(train_x),np.amax(train_x))
print (np.amin(validation_x),np.amax(validation_x))
print (np.amin(test_x),np.amax(test_x))

y = np.argmax(train_y,axis=1)
unique,counts = np.unique(y,return_counts=True)
print (np.asarray([unique,counts]))


print ("Augmenting the training dataset offline")
for disease in labels:
    number_of_generated_images = disease_counts[labels_to_index[disease]]
    print ("Augmenting for class %s with %d images" %(disease,number_of_generated_images))
    generator = generator_model[disease]
    noise = np.random.normal(0,1,(number_of_generated_images,200))
    generated_img = generator.predict(noise)
    generated_img = 0.5* generated_img + 0.5

    generated_images = np.array(generated_img)
    print (generated_images.shape)
    train_x = np.concatenate((train_x,generated_images),axis=0)

    a = np.zeros((number_of_generated_images,4))
    y = labels_to_index[disease]
    a[:,y]=1
    print (a.shape)
    print (a)
    train_y = np.concatenate((train_y,a),axis=0)


print ("Augmented dataset size")
print (train_x.shape,train_y.shape)

from sklearn.utils import shuffle
train_x,train_y = shuffle(train_x,train_y)



img_width, img_height = IMG_SIZE,IMG_SIZE
input_shape = (img_width,img_height,3)

model=Sequential()
model.add(Conv2D(16,(5,5),input_shape=input_shape,activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(160,activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(80,activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(4,activation="softmax"))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()








file_path="GAN_weights_3.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                   verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint,reduce_lr]

try:
    history = model.fit(train_x,train_y,batch_size=batch_size,epochs=HM_EPOCHS,
    validation_data=(validation_x,validation_y),callbacks=callbacks_list)
except OSError:
    pass
    #s=s+x_batch.shape[0]
    #print x_batch.shape,y_batch.shape
    #print s
with open('gan_history_3.pickle','wb') as f:
    pickle.dump(history.history,f)

gan_model = load_model(file_path)
score = gan_model.evaluate(test_x,test_y)
print ("Test Set accuracy = %f%%" %(score[1]*100))

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    fig.savefig('GAN_Training_curve_3.png')

testY = np.argmax(test_y,axis =1)

from sklearn.metrics import accuracy_score

test_pred=np.argmax(gan_model.predict(test_x),axis=1)
acc=accuracy_score(testY,test_pred)
print(acc)

#plot_model_history(history)

from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig("GAN_Confusion_Matrix_3.png")

# Predict the values from the validation dataset
Y_pred = gan_model.predict(test_x)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(test_y,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)



# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(4))

from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(Y_true, Y_pred_classes)

print(report)
