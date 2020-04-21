#!/usr/bin/env python
# coding: utf-8

# # Making a SageMaker image-classifier microservice
# 
# ## The Lambda function
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
# This is the lambda function code used by the microservice to pass received images via API posts to the model endpoint for inference. This Lambda was modified from the one presented in class to allow batches of images to be passed in one post. 
# 
# The environment variable ENDPOINT_NAME must be set to the desired model endpoint. In my case, the endpoint was "jmb-dog-breed-id-all-04-08".
# 
# 

# In[ ]:


def lambda_handler(event, context):
    """Pass one or more images to model for classification.
    input is in the form of base64 strings"""
    images = event['body']
    # if passed only one image, then make it a list of one
    if type(images) == str: images = [images]
    predictions = []
    for image in images:
        payload = get_image_payload(image)
        response = sgm_runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                               ContentType='application/x-image',
                                               Body=payload)
        predictions.append(json.loads(response['Body'].read().decode()))
    return {
        'statusCode': 200,
        'body': predictions}

