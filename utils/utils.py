import os



import glob
import cv2
import numpy as np
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
#from keras import backend as K
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import scipy.io as sio
import tensorflow as tf

from random import shuffle
import random
#import keras
from PIL import Image
#from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def trainImage_generator(files, batch_size = 32, sz = (256, 256)):
  
  while True: 
    
    #extract a random batch 
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    # batch_z = []
    
    
    for f in batch:




        #get the masks. Note that masks are png files 
        mask = cv2.imread(f'../downloads/data4test/aug_label/{f[:-3]}'+"png", cv2.IMREAD_GRAYSCALE).astype(np.float32)


        #preprocess the mask 
        # mask[mask >= 2] = 0 
        # mask[mask != 0 ] = 1
        


        #preprocess the raw images 
        # raw = Image.open(f'syntheticData/data/{f}')
        # raw = raw.resize(sz)
        # raw = np.array(raw)

        rawMat = sio.loadmat(f'../downloads/data4test/aug_data/{f}')
        xData = np.array(rawMat['vxSample'])
        yData = np.array(rawMat['vySample'])

        ImgSize = xData.shape

        input_image = np.stack((xData,yData,np.zeros(xData.shape)), -1)
        # input_mask = np.stack((mask,mask,mask), -1)


        # input_image = cv2.resize(input_image, sz)
        # input_mask = cv2.resize(mask, sz)

        # flipping random horizontal or vertical
        # if random.random() > 0.5:
        #     input_image = cv2.flip(input_image,0)
        #     input_mask = cv2.flip(input_mask,0)
        # if random.random() > 0.5:
        #     input_image = cv2.flip(input_image,1)
        #     input_mask = cv2.flip(input_mask,1)


        # input_image = np.asarray(input_image)
        # input_mask = np.asarray(mask)


        # input_mask = tf.image.rgb_to_grayscale(input_mask)

        # input_image = np.asarray(input_image)
        input_mask = mask


        input_mask[input_mask != 0 ] = 1

        batch_x.append(input_image)
        batch_y.append(input_mask)
        # batch_z.append(f)


    #preprocess a batch of images and masks 
    # batch_x = np.array(batch_x)/255.
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # batch_y = np.expand_dims(batch_y,3)

    # batch_x = preprocess_input(batch_x)
    # batch_y = preprocess_input(batch_y)



    yield (batch_x, batch_y)    





def validImage_generator(files, batch_size = 32, sz = (256, 256)):
  
  while True: 
    
    #extract a random batch 
    batch = np.random.choice(files, size = batch_size)    
    
    #variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    # batch_z = []
    
    
    for f in batch:



        #get the masks. Note that masks are png files 
        mask = cv2.imread(f'../downloads/data4test/label/{f[:-3]}'+"png", cv2.IMREAD_GRAYSCALE).astype(np.float32)


        #preprocess the mask 
        # mask[mask >= 2] = 0 
        # mask[mask != 0 ] = 1
        


        #preprocess the raw images 
        # raw = Image.open(f'syntheticData/data/{f}')
        # raw = raw.resize(sz)
        # raw = np.array(raw)

        rawMat = sio.loadmat(f'../downloads/data4test/data/{f}')
        xData = np.array(rawMat['vxSample'])
        yData = np.array(rawMat['vySample'])

        ImgSize = xData.shape

        input_image = np.stack((xData,yData,np.zeros(xData.shape)), -1)

        input_mask = mask



        input_image = np.asarray(input_image)
        input_mask = np.asarray(input_mask)
        input_mask[input_mask != 0 ] = 1

        batch_x.append(input_image)
        batch_y.append(input_mask)
        # batch_z.append(f)


    #preprocess a batch of images and masks 
    # batch_x = np.array(batch_x)/255.
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # batch_y = np.expand_dims(batch_y,3)

    # batch_x = preprocess_input(batch_x)



    yield (batch_x, batch_y)




def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(K.cast(inputs,'float32'))
    targets = K.flatten(K.cast(targets,'float32'))
    
    intersection = K.sum(targets*inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice



import tens

def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = t.flatten(K.cast(inputs,'float32'))
    targets = K.flatten(K.cast(targets,'float32'))
    
    BCE =  tf.keras.losses.binary_crossentropy(targets, inputs)
    intersection = K.sum(targets*inputs)   
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

def build_callbacks():
        checkpointer = ModelCheckpoint(filepath='./output/unet.h5', verbose=0, save_best_only=True, save_weights_only=True)
        callbacks = [checkpointer, PlotLearning()]
        return callbacks

# inheritance for training process plot 
class PlotLearning(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        #self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        print('i=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),'mean_iou=',logs.get('mean_iou'),'val_mean_iou=',logs.get('val_mean_iou'))
        
        f = open('./output/loss.txt', 'a')
        f.write(str(logs.get('loss'))+'\n')
        f.close()

        f = open('./output/val_loss.txt', 'a')
        f.write(str(logs.get('val_loss'))+'\n')
        f.close()

        f = open('./output/mean_iou.txt', 'a')
        f.write(str(logs.get('mean_iou'))+'\n')
        f.close()

        f = open('./output/val_mean_iou.txt', 'a')
        f.write(str(logs.get('val_mean_iou'))+'\n')
        f.close()




tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

def train_model(BACKBONE, model:tf.keras.models, strategy, training_data, validation_data, train_steps, test_steps):

    with strategy.scope():

        # define model
        model = sm.Unet(BACKBONE, encoder_weights='imagenet')
        for layer in model.layers:
            layer.trainable = False
        # model.compile('Adam', loss=sm.losses.bce_dice_loss, metrics=[sm.metrics.iou_score])
        model.compile('Adam', loss=sm.losses.bce_dice_loss, metrics=[sm.metrics.iou_score])

        # print(model.summary())


        model.fit(training_data,epochs = 400, steps_per_epoch = train_steps,validation_data = validation_data, validation_steps = test_steps,  callbacks = build_callbacks(), verbose = 1)

        #accuracy = model.evaluate(x_val, y_val)

    model.save('./output/newNetwork.h5')