#!/usr/bin/env python
# coding: utf-8

# # Making a SageMaker image-classifier microservice
# 
# ## Generate LST files for SageMaker model training and validation
# 
# ### Project for AICamp course: [Full Stack Deep Learning in AWS - Online](https://learn.xnextcon.com/course/courseDetails/C2031717)
# 
# ### John Burt    
# 
# #### March 2020
# 
# 
# ## The project
# For this project, the task was to create and train an image classifier using Amazon SageMaker, then deploy a classifier instance as a microservice via the AWS API Gateway. I chose to make a dog breed classifier using a set of images made available by Stanford University. 
# 
# ### For more details, see my [project github site](https://github.com/johnmburt/projects/tree/master/AWS/sagemaker_dog_breed_id)
# 
# 
# ## This notebook
# 
# This notebook generates the LST files necessary to train and test the model using SageMaker explorer. LST files describe the samples to use for training and testing the model. These files are uploaded to the S3 bucket folder that contains the train/test images and are used in the training job setup.
# 

# ### Func to read image paths from a folder and return LST data
# 
# LST data is a tab delimited text file w/ cols:
# - sample id 
# - class id 
# - sample file path (rel to root folder)
# 
# This code assumes images for each class are stored in separate subfolders in a root folder. The subfolder names are assumed to contain class name info and are returned in the dataframe so you can map class index to a class name.

# In[4]:


import pandas as pd
import numpy as np
import os
import fnmatch
from datetime import datetime,timedelta  

def read_image_lst_info(srcdir):
    """Walk through base folder and collect paths for all image files.
        category info, return as a dataframe w/ 
        samp_index, cat_index, relpath, class name"""
    
    fileexts=['*.jpg']

    # search through source folder for sample files
    relpath = []
    subdirname = []
    for ext in fileexts:
        for root, dirnames, filenames in os.walk(srcdir):
            for filename in fnmatch.filter(filenames, ext):
                subdir = root.split('\\')[-1]
                relpath.append( subdir + '/' + filename)
                subdirname.append(subdir)
                
    # make sample id
    sampid = np.arange(len(subdirname))
    
    # subdir names will be used as class names
    classnames = np.unique(subdirname)
    
    # generate class id for each sample
    d = dict(zip(classnames,np.arange(len(classnames))))
    classid = [d[x] for x in subdirname]
    
    # return dataframe with file info
    return pd.DataFrame({'sampid': sampid, 
                         'classid':  classid,
                         'path': relpath,
                         'classname': subdirname} )   
    
# dir containing image files
# NOTE: code assumes this script is run from directory 
#  containing srcdir.
srcdir = './images'

df = read_image_lst_info(srcdir)


# ### Generate train/test split(s), save to LST files
# 
# - Optionally, select a subset of classes: for model development and tuning with a large number of classes, it's faster to work with a subset initially, then train on the full set after you're satisfied with subset performance.
# 
# 
# - Shuffle samples, stratify to ensure sample frequencies are balanced, split into train and test sets. 
# 
# 
# - If more than one split (n_splits>1), generate multiple sets of train/test samples, using K folds method.
# 

# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit

# num K folds train/test split sets you want to generate
n_splits = 1 

# None = select all classes
n_classes = None 

# prefix for generated LST file names
filenameroot = 'dog_breeds'

# include all classes of image
if n_classes is None:
    filenameroot += '_all'
    df_filt = df
# include only the first specified number of classes
else:
    filenameroot += '_'+str(n_classes)
    df_filt = df[df.classid<n_classes]

# this split method ensures each class gets equal sample sizes
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

print('Generating train/test lst files:')
for i, (train_index, test_index) in zip(range(n_splits), 
                    sss.split(df_filt, df_filt['classid'])):
    df_train = df_filt.iloc[train_index,:3]
    fname_train = filenameroot+'_fold_%d_train.lst'%(i+1)
    df_train.to_csv(fname_train, index=False, header=False, sep='\t')
    
    df_test = df_filt.iloc[test_index,:3]
    fname_test = filenameroot+'_fold_%d_test.lst'%(i+1)
    df_test.to_csv(fname_test,index=False, header=False, sep='\t')
    
    print('split',i+1)
    print('  train: %d samples, %d classes, %s'%(
        df_train.shape[0], len(df_train.classid.unique()), fname_train))
    print('  test: %d samples, %d classes, %s'%(
        df_test.shape[0], len(df_test.classid.unique()), fname_test))

