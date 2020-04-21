#!/usr/bin/env python
# coding: utf-8

# 
# # Making a SageMaker image-classifier microservice
# 
# ## Get sample info on a generated LST file set
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
# This notebook can be used to help set up hyperparams for a Sagemaker training job. The model will want to know number of clases, and (max) number of samples (120 categories, 20580 images).
# 

# In[1]:


import pandas as pd
import numpy as np

n_splits = 1
n_classes = None

filenameroot = 'dog_breeds'

# include all classes of image
if n_classes is None:
    filenameroot += '_all'
# include only the specified number of classes
else:
    filenameroot += '_'+str(n_classes)

# Create filenames to read
fname_train = filenameroot+'_fold_1_train.lst'
fname_test = filenameroot+'_fold_1_test.lst'

# read previously generated train and test data 
df_train = pd.read_csv(fname_train, sep='\t', 
                 names=['sampid','classid','path'])
df_test = pd.read_csv(fname_test, sep='\t', 
                 names=['sampid','classid','path'])

print('  train: %d samples, %d classes, %s'%(
    df_train.shape[0], len(df_train.classid.unique()), fname_train))
print('  test: %d samples, %d classes, %s'%(
    df_test.shape[0], len(df_test.classid.unique()), fname_test))


# In[ ]:




