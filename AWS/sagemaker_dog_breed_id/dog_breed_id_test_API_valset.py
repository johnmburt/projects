#!/usr/bin/env python
# coding: utf-8

# # Making a SageMaker image-classifier microservice
# 
# ## Analyze classifier performance via the API
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
# At this stage, my model has been trained and is running as a web microservice via the AWS API Gateway. Now, I want to examine the performance of the model in more detail. The model has 83% validation accuracy overall, but is that uniform? In particular, I need to know if some classes perform poorly, and if so try to understand why they do.  
# 
# 
# ### Results summary
# 
# - The classifier performed very well overall, with most classes at > 80% recall (#_correct_class_N / #_samples_class_N).
# 
# 
# - There were low performing classes. The lowest had a recall of ony 3%.
# 
# 
# - Looking at breed confusion errors helped explain why some classes had low recall. Low performing classes appeared to have breed labelling errors (Husky labelled as Eskimo Dog, etc), were the result of the same breed split into multiple size classes (Poodle) or were different breeds that look very similar (Lhasa Apso, Maltese Dog, Shi Tzu).
# 
# 

# ## Read the test LST file
# 
# The test LST file gives me the paths of the validation images I'll use for testing. I can also parse dog breed names for each class from the path.

# In[20]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

import pandas as pd
pd.options.display.max_columns = 100

import numpy as np
import seaborn as sns

# lst file used for testing
testlstfile = 'dog_breeds_all_fold_1_test.lst'

df = pd.read_csv(testlstfile, sep='\t', names=['sampid','classid','path'])
classnames = np.array([s.split('-')[1].split('/')[0] 
                       for s in df.groupby(by='classid').first().path])


# ## Load dog images, classify breeds
# 
# I want to pass all of the test images to the classifier, but due to potential bandwidth issues, I break the test data into sets of images for each breed and send each with one API call to the classifier.
# 
# From the classifier results, I create two confusion matrices: counts based on max output class, and mean output of all test samples for each class.
# 
# Note: I've modified the Lambda function to take multiple images in the form of a list. The API then returns a list of results.

# In[34]:


import base64 # encode/decode image in base64
import json
import requests

def encode_image(filepath):
    """read the image, convert to base64, embed in json payload object"""
    with open(filepath, 'rb') as image_file:
       return base64.b64encode(image_file.read()).decode('utf-8')

# test image root folder
rootdir = './images/'

# my breed prediction microservice API URL
api_url = 'https://uqquh6whab.execute-api.us-west-2.amazonaws.com/beta/classify'

# confusion matrix for mean output values
cm_mean = pd.DataFrame(np.zeros((len(classnames),len(classnames))))

# confusion matrix for max output counts
cm_count = pd.DataFrame(np.zeros((len(classnames),len(classnames))))

# Pass sets of images to the classifier, ordered by dog breed
for i, name in zip(range(len(classnames)),classnames):
    print('testing %3d: %s'%(i,name), end='')
    enc_strings = [encode_image(rootdir+path) for path in 
                   df[df.classid==i]['path']]
    print(' %d test images'%(len(enc_strings)))
    payload = json.dumps( {'body': enc_strings } )
    # post the image, receive inference response
    r = requests.post(url=api_url, data=payload, timeout=50)
    # Classifier output is a list of results,
    classouts = np.array(r.json()['body'])
    for out in classouts:
        cm_count.iloc[i, np.argmax(out)] += 1
        cm_mean.iloc[i, :] += out
    cm_mean.iloc[i, :] /= len(enc_strings)
        


# In[161]:


def plot_similarity_mx_df(similarity_mx, title='', show_labels=True,
                          plot_colorbar=True):
    """Display a similarity matrix as a square colored table 
    with color darkness indicating degree of similarity between
    each comparison pair.
    """
    ax = plt.gca()
    cax = ax.matshow(similarity_mx, cmap=plt.cm.Blues)
    if plot_colorbar:
        plt.gcf().colorbar(cax,fraction=0.046, pad=0.04)
    if show_labels:
        ax.set_xticks(range(len(similarity_mx.columns)))
        ax.set_yticks(range(len(similarity_mx.columns)))
        ax.set_xticklabels(similarity_mx.columns, rotation = 90)
        ax.set_yticklabels(similarity_mx.columns)
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(False)
    plt.title(title)


# ## All classes confusion matrix
# 
# The confusion matrix for all classes looks pretty good: the classifier correctly identifies test samples for most breeds. 

# In[183]:


plt.figure(figsize=(7, 5))
plot_similarity_mx_df(cm_count, title='Classifier confusion matrix - all classes', 
                      show_labels=False, plot_colorbar=True)


# ## Recall for each class
# 
# - Looking at recall (#correct_class_n / #samples_class_n), I see that the majority of classes are above 80%. 
# 
# 
# - There is a sharp decline in recall below about 55%, where 6 classes perform particularly badly, and one, "eskimo_dog", has only 3% recall.
# 
# 
# ### What's going on with the low-recall classes?
# 
# I want to look at these low performing classes more closely.

# In[176]:


print('\nLowest accuracy')
print('\nID\trecall\tbreed name\n')
for i in idx[:15]:
    print('%d\t%1.2f\t%s'%(i,recall[i],classnames[i]))
    


# In[184]:


nclasses = cm_count.shape[0]

recall = np.zeros(nclasses,)

# calculate recall for each class
for i in range(nclasses):
    recall[i] = cm_count.iloc[i,i]/np.sum(cm_count.iloc[i,:])

idx = np.argsort(recall)

fig = plt.figure(figsize=(15,5))
plt.bar(range(nclasses), recall[idx]);
plt.ylabel('Class recall')
plt.xlabel('Class ID')
plt.title('Image classification recall for 120 dog breeds', fontsize=20);


# ## Looking at breeds with lowest recall
# 
# For this analysis I group the 5 lowest recall breeds with the 2 other breeds that the classifier tends to falsely predict for them the most. I then visualize the confusion matrix for each set to see what seems to be wrong.
# 
# ### Results:
# 
# - The worst recall breed class is Eskimo Dog, which is most often confused with Malamute and Siberian Husky. The images for these three breeds are very similar in appearance, and in fact comparing images at Eskimo Dog websites to the train/test images, it's clear that most of the dogs tagged as "Eskimo Dog" are in fact Huskies or Malamutes. So, this is appears to be a case of mislabelling.
# 
# 
# - The next low recall breed is Walker Hound, which is confused mostly with English Foxhound and occasionally with Basset. In this case, the two breeds just look very similar. www.coonhoundscompanions.com writes: 
# 
#     >"It’s pretty hard to tell the difference between Treeing Walker coonhounds and foxhounds. Most of the differences are behavioral rather than visibly structural, and the behavioral differences are most evident in a hunting setting."
#     
# 
# - Collies are confused with close relatives Border Collies and Shetland Sheepdogs. 
# 
# 
# - Miniature Poodles are confused with Toy and Standard Poodles. Here is a case of one breed being separated by size, which confuses the classifier since it doesn't know about size.
# 
# 
# - Lhasas are confused with Maltese Dogs and Shihs, which look very similar comparing images.

# In[177]:


# number of low recall classes to analyze
nworst = 5
# number of highest count classes for each low recall class to collect.
# note: the highest count class will probably be the class itself.
nclosest = 3

worst = idx[:nworst]

# collect list of worst recall IDs and the ids of breeds the classifier 
#  confused them with
idsets = []
idnames = []
for i in worst:
    # exclude ids that already exist in prev set,
    #  so like groupings will only be shown once.
    if not idsets or i not in idsets[-1]:
        # collect id for this class plus classes confused w/ it
        idset = [i]
        idset.extend(np.argsort(-cm_count.iloc[i,:])[:nclosest])
        # use pandas to remove dupes w/out sorting
        idsets.append(pd.Series(idset).unique()[:nclosest])
        idnames.append(classnames[idsets[-1]])


# In[181]:


import matplotlib.image as mpimg

def plot_image(imgpath):
    """read image file and display it in the current figure"""
    img = mpimg.imread(imgpath)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.grid(b=None)

# show details about classes w/ the lowest recall
for ids, names in zip(idsets,idnames):
    # create offset breed name title
    plt.figure(figsize=(15, 1));
    plt.axis('off')
    plt.grid(b=None)
    plt.title(names[0], fontsize=24, y=-0.01)
    
    # draw a confusion mx for just these breeds
    wcm = cm_count.iloc[ids,ids]
    # set cols to breed names for figure tick labels
    wcm.columns = names
    plt.figure(figsize=(6, 4))
    plot_similarity_mx_df(wcm, title='Breed class confusions', plot_colorbar=True)
    
    # show example images of the breeds 
    plt.figure(figsize=(15, 4))
    for i, bid, name in zip(range(len(ids)),ids,names):
        plt.subplot(1,len(ids),i+1)
        plot_image(rootdir+df.path[df.classid==bid].values[11])
        plt.title(name,fontsize=19)
        plt.gca().set_anchor('N');

