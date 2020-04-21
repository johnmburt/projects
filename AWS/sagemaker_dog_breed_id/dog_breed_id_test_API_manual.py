#!/usr/bin/env python
# coding: utf-8

# # Making a SageMaker image-classifier microservice
# 
# ## Manually test the API and model function by sending a single dog image
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
# Test the trained model instance, via an AWS Gateway API call. This notebook lets you select a local image file, prepares the image data, passes it to the model for inference, then displayes results. 
# 

# ## Get class info from training lst file 
# 
# I'll use the directory name in the path to get dog breed name for each class.

# In[33]:


import pandas as pd
import numpy as np

# lst file used for training
trainlstfile = 'dog_breeds_all_fold_1_train.lst'

df = pd.read_csv(trainlstfile, sep='\t', names=['sampid','classid','path'])
classnames = np.array([s.split('-')[1].split('/')[0] 
                       for s in df.groupby(by='classid').first().path])


# ## Filename selection dialog

# In[34]:


get_ipython().run_line_magic('gui', 'qt')

from PyQt5.QtWidgets import QFileDialog

def gui_fname(dir=None, filters=None):
    """Select a file via a dialog and return the file name."""
    if dir is None: dir = './'
    if filters is None: filters = 'All files (*.*)'
    fname = QFileDialog.getOpenFileName(None, "Select file...", 
                dir, filter=filters)
    return fname[0]


# ## Select dog image, classify breed
# 
# - Use a file selection dialog to choose a dog image file in a local folder.
# 
# - Format and embed image into json payload object.
# 
# - Post to the image payload to the API gateway.
# 
# - Receive model results and select breed ID based on highest output value.
# 
# Note: I've modified the Lambda function to take multiple images in the form of a list. The API then returns a list of results. If I send a single image in the payload, the Lambda detects this and just does one classification, but still returns a list of results with one element.

# In[46]:


import base64 # encode/decode image in base64
import json
import requests

# Collection of dog images not in training set
rootdir = 'C:/Users/john/notebooks/aicamp/dogs/test_images'

# select an image file to test
imgpath = gui_fname(dir=rootdir, filters='image (*.jpg)')

# read the image, convert to base64, embed in json payload object
with open(imgpath, 'rb') as image_file:
   encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
payload = json.dumps( {'body': encoded_string } )

# my breed prediction microservice API URL
api_url = 'https://uqquh6whab.execute-api.us-west-2.amazonaws.com/beta/classify'

# post the image, receive inference response
r = requests.post(url=api_url, data=payload, timeout=5)

try:
    # Classifier output is a list of results,
    # Since I sent one image, it will be the first list element
    classout = np.array(r.json()['body'])[0]

    # select highest output 
    selclass = np.argmax(classout)
    # sort by output, desc
    sortidx = np.argsort(-classout)
    print('Predicted dog breed, sorted by model output:')
    print('\nID\tOutput\tBreed name\n')
    for i, x, classname in zip(sortidx,classout[sortidx],classnames[sortidx]):
        print('%d   \t%1.3f %s\t%s'%(i,x,'*' if selclass==i else ' ', classname))
except:
    print('There was an error accessing the API.')
    print('Check:')
    print('  - Model endpoint is In Service')
    print('  - Lambda is using correct endpoint and is updated')
    print('  - API is active and updated')
    

