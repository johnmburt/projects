#!/usr/bin/env python
# coding: utf-8

# # Ancient Chinese characters
# 
# ## A Chinese character code classifier
# 
# ### John Burt
# 
# #### November 2018
# 
# The Chinese character data used in this notebook was provided to participants in the November  2018 Applied Data Science meetup series. From the meetup intro:
# 
# >Our data for November and December is a simple entry point to image analysis: automated recognition of historical, handwritten, Chinese characters, some of which may date back to over 3,000 years ago.
# 
# The dataset consisted of a set of .PNG grayscale character image files. Filenames contained the character's ID code, and the "era" (era actually refers to the time period and the source of the images - paper, pottery, etc). There were usually multiple examples for each character code and era, but the sample number varied. In this example, I only considered the character code for classification and included samples of all "eras" for training and testing.
# 
# An unlabelled holdout dataset was provided as a challenge to those who wanted to build a classifier to predict their character code and/or era.
# 
# 
# ### My goal for this notebook: 
# - Build and train a CNN to classify chinese character images to character code.
# - Use the trained model to predict ID codes for the unlabelled holdout set
# 
# #### Methods:
# - Read PNG image file directories, parse filenames to create pandas dataframe with sample info.
# 
# 
# 
# - Generate balanced training data:
#     - Separate train and test samples
#     
#     - Upsample using duplicates so that each category has the same number of training samples
# 
#     - Shuffle train and test datasets
# 
# 
# 
# - Create a CNN - based multilayer model. The inspiration for this model came from [this article](http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/). Given the simplicity of this dataset, a convolutional network is probably overkill, but I wanted to explore implementing a CNN.
# 
# 
# 
# - Train using randomly modified training images
# 
# 
# 
# - Generate and save test data predictions
# 
# 
# #### Extra packages required:
# - scikit-image
# - pil
# - keras
# 

# In[1]:


# basic notebook setup

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np


# ### Parse datafiles to create sample info dataframe
# 
# This sample info parsing is a bit complicated because I have to traverse multiple subfolders and read filenames from each.

# In[2]:


import os
import re

def get_image_info(sourcetop):

    codestr = [] # character code (converted to int)
    code = [] # character code (converted to int)
    era = [] # era
    index = [] # index of sample for this code and era
    path = [] # file path to read image data

    # Seek through data folder for PNG files
    for dirpath, dirnames, filelist in os.walk(top=sourcetop):
        # change any \ to /
        srcdir = dirpath.replace('\\','/')
        # look for any PNG files in the current folder
        for fname in filelist:
            if fname[-3:]=='png':
                # parse filename to get image category data
                parts = re.split('/|-|_|\.',fname)
                # files with category info in filename
                if len(parts) > 2:
                    # get image file path
                    path.append(srcdir+'/'+fname)
                    codestr.append(parts[0]) # character code str
                    code.append(int(parts[0], 16)) # convert character code str to int
                    era.append(parts[1])
                    index.append(int(parts[2]))  
                # modern figure image file
                else:
                    path.append(srcdir+'/'+fname)
                    codestr.append(parts[0]) # character code str
                    code.append(int(parts[0], 16)) # convert character code str to int
                    era.append('modern')
                    index.append(0)    

    # create info dataframe
    df = pd.DataFrame( 
        {
        'codestr' : codestr, 
        'code' : code, 
        'era' : era, 
        'id' : index, 
        'path' : path
        } )
    
    return df


sourcetop = 'ch_train_set' # name of dir holding image data

df = get_image_info(sourcetop)

# df.head()
print(df.shape)


# ### Load image data
# 
# - Load image using PIL 
# - Convert to numpy arrays of 1/0 data
# - Add a pixel buffer of 0s around image to allow rotation and shifting without clipping
# 

# In[3]:


from PIL import Image

def load_images(df, pixelbuffer=0):
    images = []
    labels = (df.codestr + '_'  + df.era).values
    numimages = df.shape[0]
    for path,i in zip(df.path,range(numimages)):
        # read image, convert to int, and invert so that 1=ink, 0=whitespace
        arr = np.asarray(Image.open(path)).astype(int)
        if type(images) == list:
            xdim = arr.shape[0]
            ydim = arr.shape[1]
            images = np.zeros( (numimages, xdim+pixelbuffer*2, ydim+pixelbuffer*2 ) )
        images[i,pixelbuffer:pixelbuffer+xdim, pixelbuffer:pixelbuffer+ydim] = arr
    return images, labels

pixelbuffer = 20

print('loading images...')
images, labels = load_images(df, pixelbuffer=pixelbuffer)
print('  done')


# ### Generate balanced training / testing data:
# 
# - (optional) Select only characters that have a specified minimum number of examples in each era/script. 
# - Separate train and test images
# - Upsample using duplicates so that each category has the same number of output training samples
# - Shuffle train and test datasets
# 

# In[5]:




def create_balanced_traintest_data(df, images, labels,
                                   minsamples=[50, 0, 0, 0], outputsamples=[200,0], 
                                   proptest=0.2 ):
    
    def sample_up(arr, numsamps):
#         print('sample_up arr.shape',arr.shape,'numsamps',numsamps)
        if arr.shape[0] >= numsamps:
            arrout = arr[:numsamps,...]
        else:
            numrepeats = int(numsamps/arr.shape[0])
            lastsize = numsamps%arr.shape[0]
            arrout = arr.copy()
            for i in range(numrepeats-1):
                arrout = np.append(arrout, arr, axis=0)
            if lastsize:
                arrout = np.append(arrout, arr[:lastsize,...], axis=0)                                
        return arrout
        
    # get output era categories 
    eras = ['chuxi', 'jinwen', 'oracle', 'smallseal']
    outeras = []
    outeras_idx = []
    for minsamp, era, i in zip(minsamples,eras,range(len(eras))):
        if minsamp >= 0 : 
            outeras.append(era)
            outeras_idx.append(i)
            
    print('output eras = ', outeras)
 
    # find the character codes that meet min count threshold 
    okcodes = df.code.unique() # get list of codes
    
    print("\n%d codes met min criteria of "%(len(okcodes)),minsamples)
    
    # collect samples of each code
    #   split into train/test sets and split X and y
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]

    for code in okcodes:
        # collect samples from each output era
        for era, eraidx in zip(outeras, outeras_idx):
            # get numbers of samples to pull for train/test sets
            numsamps = df[(df.code==code) & (df.era==era)].shape[0]
            if numsamps == 0:
                print('no samples %X %s'%(code,era))
            elif numsamps < minsamples[eraidx]:
                print('not enough samples  %X %s %d < %d'%(code,era,numsamps, minsamples[eraidx]))
            else:
                numtest = int( numsamps * proptest )
                if numtest < 1: numtest = 1
                numtrain = numsamps - numtest
                if numtrain < 1: numtrain = 1
                # get images
                X = images[(df.code==code) & (df.era==era),...]
                y = labels[(df.code==code) & (df.era==era)]
                # get indices of samples within whole dataset
                idx = list(range(y.shape[0]))
                # shuffle indices
                np.random.shuffle( idx )
                # if this is the first time, create the output dfs
                if type(X_train) == list:
                    X_train = sample_up(X[idx[:numtrain],...], outputsamples[0])
                    X_test  = sample_up(X[idx[-numtest:],...], outputsamples[1])
                    y_train = sample_up(y[idx[:numtrain]], outputsamples[0])
                    y_test  = sample_up(y[idx[-numtest:]], outputsamples[1])
                else:
                    X_train = np.append(X_train, sample_up(X[idx[:numtrain],...], outputsamples[0]), axis=0)
                    X_test  = np.append(X_test, sample_up(X[idx[-numtest:],...], outputsamples[1]), axis=0)
                    y_train = np.append(y_train, sample_up(y[idx[:numtrain]], outputsamples[0]), axis=0)
                    y_test  = np.append(y_test, sample_up(y[idx[-numtest:]], outputsamples[1]), axis=0)

                print('code %x, era %s, numsamps=%d, numtrain=%d, numtest=%d'%(
                    code,era,numsamps,numtrain,numtest))
                
    return X_train, y_train, X_test, y_test


# ### If there is a train/test set saved to disk, then load it, otherwise generate a new one
# 
# Generating a balanced train / test set takes a while. Much quicker to load a version that's already been created and saved to disk.

# In[7]:


import pickle

# note: -1 = exclude that era
minsamples = [1, 1, 1, 1] # collect all characters, even if there's only 1 example
outputsamples = [200, 20] # output 200 samples of each character for training, 20 for testing

modelname = 'char_traintest_set_v1'

try:
    with open(modelname+'.pkl', 'rb') as handle:
        X_train = pickle.load(handle)
        y_train = pickle.load(handle)
        X_test = pickle.load(handle)
        y_test = pickle.load(handle)
        
    print('train and test data loaded from file',modelname+'.pkl')
    
except:
    X_train, y_train, X_test, y_test = create_balanced_traintest_data(df, images, labels,
                                            minsamples=minsamples,
                                            outputsamples=outputsamples,
                                            proptest=0.2)
    with open(modelname+'.pkl', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('train and test data created and saved')
    
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ### Transform the image data for NN training and testing
# 
# - Create 4D input arrays, containing a single color channel as the 4th dimension.
# - Convert character class and era into 1-hot output vector.

# In[8]:


from sklearn import preprocessing
import keras

# input image dimensions
img_x, img_y = X_train.shape[1], X_train.shape[2]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the images are greyscale, we only have a single channel - RGB colour images would have 3
X_train_rs_orig = X_train.reshape(X_train.shape[0], img_x, img_y, 1)
X_test_rs_orig = X_test.reshape(X_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert text class labels into index numbers
classnames = np.unique(y_train)
le = preprocessing.LabelEncoder().fit(classnames)
y_train_int = le.transform(y_train)
y_test_int = le.transform(y_test)

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
num_classes = np.unique(y_train).shape[0]
y_train_bin = keras.utils.to_categorical(y_train_int, num_classes)
y_test_bin = keras.utils.to_categorical(y_test_int, num_classes)


# ### Modify training images during training
# 
# Durint training, this is called to randomly shift and/or rotate training images to help the network learn to generalize better.
# 

# In[9]:


import PIL
from random import random

def modify_images(images, 
                  rotate=0, shift=[0,0], 
                  stretch=[0,0], skew=[0,0]):
    """Modify images by shifting anr/or rotating them"""
    
    modimages = images.copy()
    
#     print('modimages.shape',modimages.shape)
    
    for image in modimages:
#         print('image.shape',image.shape)
        pim = PIL.Image.fromarray(image[:,:,0])
        
        if rotate > 0:
            pim = pim.rotate(random() * rotate*2 - rotate)
            
        if shift[0] > 0:
            pim = PIL.ImageChops.offset(pim, 
                    int(random()*shift[0]*2-shift[0]), 
                    int(random()*shift[1]*2-shift[1]))
            
        image[:,:,0] = np.asarray(pim).astype(int)
        
    return modimages


# ### Create convolutional neural net
# 
# Based in part on code from:
# 
# - [Keras tutorial – build a convolutional neural network in 11 lines](http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/)
# - [Tutorial code](https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_cnn.py)
# 
# #### Model layer structure:
# - Convolution X 64
# - Dropout
# - Max pooling
# - Convolution X 128
# - Dropout
# - Max pooling
# - Dense X 1000
# - Dropout
# - Output
# 
# 
# #### image modification during training:
# - Model training is iterated for several epochs, then the training images are rotated and shifted randomly. This helps the model generalize (based on tests done in a different notebooks script).

# In[10]:


import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

dropoutrate = .4
# convlayersize = [32, 64]
convlayersize = [64, 128]

adddropout = True

model = Sequential()
model.add(Conv2D(convlayersize[0], kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
if adddropout:
    model.add(Dropout(dropoutrate))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(convlayersize[1], (5, 5), activation='relu'))
if adddropout:
    model.add(Dropout(dropoutrate))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
if adddropout:
    model.add(Dropout(dropoutrate))
model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())


# ### Callback function to display performance stats during training

# In[11]:



from keras.callbacks import Callback
from IPython.display import clear_output

# Fancy updating plot callback with loss and accuracy

class PlotLearning(Callback):
    
    # ***********************************************
    def __init__(self, numepochs=0, numsamps=0, 
                 batchsize=0, updaterate=1, figsize=(15,5)):
        self.numepochs = numepochs
        self.numsamps = numsamps
        self.batchsize = batchsize
        self.updaterate = updaterate
        self.figsize = figsize
    
    # ***********************************************
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.valx = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
        self.batchnum = 0
        self.epochnum = 0
        
        self.t0 = time()

        self.batchperepoch = (int(self.numsamps/self.batchsize) + 
            int(self.numsamps%self.batchsize > 0) )
        
        self.fig = plt.figure();
        
    # ***********************************************
    def plot_log_output(self):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize);
        clear_output(wait=True)
        plt.suptitle('epoch %d / %d, batch %d / %d'%(self.epochnum+1, self.numepochs,
                                                self.batchnum+1, self.batchperepoch))
        ax1.set_yscale('log')
#             ax1.set_ylim(bottom=0)
        ax1.set_title('loss')
        ax1.plot(self.x, self.losses, 'r-', label="loss = %1.3f"%(self.losses[-1]))
        if len(self.val_losses) > 0:
            ax1.plot(self.valx, self.val_losses, 'b-x', 
                     label="validation loss = %1.3f"%(self.val_losses[-1]))
        ax1.legend()

        ax2.set_title('accuracy')
        ax2.set_ylim(bottom=0)
        ax2.plot(self.x, self.acc, 'r-', label="accuracy = %1.3f"%(self.acc[-1]))
        if len(self.val_acc) > 0:
            ax2.plot(self.valx, self.val_acc, 'b-x', label="validation accuracy = %1.3f"%(self.val_acc[-1]))
        ax2.legend()

        detailtext = ''
        for layer,i in zip(self.model.layers, range(len(self.model.layers))):
            cfg = layer.get_config()
            if 'units' in cfg.keys():
                detailtext += '  %s: %d units, %s\n'%(
                    cfg['name'], cfg['units'], cfg['activation'])
            else:
                detailtext += '  %s\n'%(cfg['name'])

        ax3.set_title('model details')
        ax3.set_xlim([0, 10])
        ax3.set_ylim([0, 10])
        ax3.text(0,9,detailtext, ha='left',va='top')
        ax3.axis('off')

        plt.show();
            
    # ***********************************************
    def on_batch_end(self, batch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.i += 1
        self.batchnum = batch
        
        # use timing to decide when to plot
#         if self.i % self.batchperupdate == 0:
        if time() - self.t0 >= self.updaterate:
            self.plot_log_output()
            self.t0 = time()

    # ***********************************************
    def on_epoch_begin(self, epoch, logs={}):   
        self.epochnum = epoch
        self.batchnum = 0
        self.t0 = time()
        
    # ***********************************************
    def on_epoch_end(self, epoch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        if 'val_loss' and 'val_acc' in logs:
            self.val_losses.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))
            self.valx.append(self.i)
        self.i += 1
#         self.plot_log_output()
    


# ### Train the model
# 
# - Iterate training several times
# - Every time rotate and shift every image randomly so that no two images are exactly the same

# In[21]:


from keras.callbacks import EarlyStopping
from time import time

t0 = time()

rotate=20
shift=[10,10]
stretch=[0,0]
skew=[0,0]

batch_size = 256
epochs = 10
updaterate = 5
    
plot_losses = PlotLearning(numepochs=epochs, 
                           numsamps=X_train_rs_orig.shape[0], 
                           batchsize=batch_size, 
                           updaterate=updaterate)

# run a few iterations
for i in range(4):
    X_train_rs = modify_images(X_train_rs_orig, rotate=rotate, shift=shift, stretch=stretch, skew=skew)
    X_test_rs = modify_images(X_test_rs_orig, rotate=rotate, shift=shift, stretch=stretch, skew=skew)
    
    model.fit(X_train_rs, y_train_bin ,
                validation_data = (X_test_rs, y_test_bin),
                callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001), 
                plot_losses])


# ### Predict character IDs for holdout images
# 
# - Prep holdout data for prediction
# - Generate prediction from trained model

# In[25]:


def get_test_image_info(sourcetop):

    index = [] # index of sample for this code and era
    path = [] # file path to read image data

    # Seek through data folder for PNG files
    for dirpath, dirnames, filelist in os.walk(top=sourcetop):
        # change any \ to /
        srcdir = dirpath.replace('\\','/')
        # look for any PNG files in the current folder
        for fname in filelist:
            if fname[-3:]=='png':
                # parse filename to get image category data
                parts = re.split('/|-|_|\.',fname)
                # files with category info in filename
                if len(parts) > 2:
                    # get image file path
                    path.append(srcdir+'/'+fname)
                    index.append(int(parts[1]))  

    # create info dataframe
    df = pd.DataFrame( 
        {
        'codestr' : ['0'] * len(index), 
        'era' : ['x'] * len(index), 
        'id' : index, 
        'path' : path
        } )
    
    return df

sourcetoptest = 'ch_hold_out/character_classification' # name of dir holding image data

# load image file info
dftest = get_test_image_info(sourcetoptest)

# load images
testim, testlabel = load_images(dftest, pixelbuffer=pixelbuffer)

img_x, img_y = testim.shape[1], testim.shape[2]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the images are greyscale, we only have a single channel - RGB colour images would have 3
testim_rs = testim.reshape(testim.shape[0], img_x, img_y, 1)

# for im in testim_rs:
#     plt.figure(figsize=(2,2));
#     plt.imshow(im[:,:,0]);

# generate predicted character IDs for holdout images
y_pred_test = np.argmax(model.predict(testim_rs),1)


# ### Transform prediction output into character codes, and save to csv for submission

# In[27]:


predictname = 'JMB_prediction_char_v2.csv'

catnames = np.unique(y_train)
codes = np.array([catnames[i].split('_')[0] for i in range(catnames.shape[0])])

pd.DataFrame(codes[y_pred_test], columns=['predict']).to_csv(predictname)


# ### Save trained model, in case we want to load it and use it later

# In[29]:


model.save(predictname+'.h5')
print('saved model ',predictname+'.h5')
    

