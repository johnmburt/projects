# Training a SageMaker Deep Learning model and deploying it as a microservice

## How I trained an image classifier model to predict dog breeds entirely from Amazon’s AWS Console website.

### John Burt
#### May 2020

![Dog mosaic](https://github.com/johnmburt/johnmburt.github.io/blob/master/images/projects/dog_mosaic.png)

Imagine you’re an app developer, and you want to make a mobile app that tells users the breed of a dog just by pointing their phone at it and taking a picture. This seems like a pretty cool idea, and it could be applied to any number of other things people might want to identify like birds, flowers, cars, etc. The app would need a classifier to predict dog breed from an image, and the best models for that are Deep Learning neural networks like ResNet. But DL models are large and processing intensive, so you should host your dog breed classifier in the cloud where the mobile app can access it via an API. 


Luckily for you, AWS SageMaker makes it incredibly easy to build, train and tune machine learning models in the cloud. Once you have a trained model it’s a simple step to implement a scalable inference microservice using the AWS API Gateway. In this post I’ll take you through the basic steps for how to do this using a project I created for AICamp’s class Full Stack Deep Learning in AWS. For this project I did everything from the AWS Management Console. 

My project github site is here: [Dog breed identification from images](https://github.com/johnmburt/projects/tree/master/AWS/sagemaker_dog_breed_id), and it includes more details about how to set up your inference microservice on AWS.

### The data

If I want to build a Deep Learning model to ID dog breeds, then I’m going to need need a lot of labeled dog images. For this I chose the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), a set of 20580 images of 120 dog breeds, with 100-150 image samples for each breed. SageMaker expects the model training data to be in the cloud, so I uploaded the images to an S3 bucket, placing them into a folder “dogs”. During model training and testing, I set up my SageMaker training job to access this folder.


### The model

SageMaker offers a built-in [image-classifier, which is a ResNet deep learning model](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html). ResNet is a large convolutional neural network, and would normally need a lot more training images than the Stanford Dogs dataset provides, but [SageMaker offers the option of transfer learning](https://docs.aws.amazon.com/sagemaker/latest/dg/IC-HowItWorks.html), which greatly reduces the number of training images required as well as training time. Transfer learning is a procedure where the network is pre-trained to a generic classification task with a very large image dataset (in this case [ImageNet](http://www.image-net.org/)), and then you re-train the model using your smaller more specific training set. 


Another training feature that SageMaker provides is [early stopping](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-early-stopping.html), where the training job will stop if the model loss metric doesn’t improve or gets worse. Early stopping and transfer learning greatly reduces training times, which also results in lower compute costs per job.

### Hyperparameter tuning

You can read about all the SageMaker image classifier [hyperparameters here](https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html). I tried several different resnet model sizes, and 50 layers seemed to be the minimum size that worked well so I chose that. I tried using [hyperparameter tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html) for several other hyperparameters, but ended up canceling the training job because it was eating up processing time and costing me too much for not much improvement. I settled on the following hyperparameters based on defaults and manual tweaking. I found that learning rate was the most important parameter to ensure good training: too large and the model would often never train properly, but too small and the model would take longer than necessary to train. I used a learning rate of 0.0001 here, but I think the ideal rate for this task is actually around 0.0005. I also enabled image augmentation during training, where the training images are randomly cropped, rotated, sheared and color modified. This reduces overfitting and helps the model to generalize better.

#### After tuning and testing, I chose the following hyperparameters:

|Key|Value| |Key|Value|
|-:|:-|--|-:|:-|
|augmentation_type|crop_color_transform| |mini_batch_size|32|
|beta_1|0.9| |momentum|0.9|
|beta_2|0.999|  |multi_label|0|
|checkpoint_frequency|1| |num_classes|120|
|early_stopping|true| |num_layers|50|
|early_stopping_min_epochs|5| |num_training_samples|16464|
|early_stopping_patience|5| |optimizer|sgd|
|early_stopping_tolerance|0| |precision_dtype|float32|
|epochs|120| |use_pretrained_model|1|
|eps|1e-8| |use_weighted_loss|0|
|gamma|0.9| |weight_decay|0.0001|
|image_shape|3,224,224| | | |
|learning_rate|0.0001| | | |
|lr_scheduler_factor|0.1| | | |


### The training job

For the model training job I needed to specify the class of each image and which images to assign for training and testing (validation). To do that, I [generated a set of tab delimited text files](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_classifier_gen_LST.ipynb), one for training and one for testing ([dog_breeds_all_fold_1_train.lst](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breeds_all_fold_1_train.lst) and [dog_breeds_all_fold_1_test.lst](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breeds_all_fold_1_test.lst)), that listed the class ID and path of each training image (training and testing image sets were mutually exclusive). These LST files were placed into the dogs folder on the s3 bucket used for model training. 


From the AWS SageMaker Studio console, I created a training job, selecting the image classifier model and configuring the hyperparameters as above, telling the job where to find the images and the LST files, and specifying a few additional configurations. The training job took 2.7 hrs (costing around $7). The final overall validation accuracy was 83%, which is not too bad, though something we'd want to improve on later.

### Creating the Deep Learning inference microservice

After training the model, I needed to get it running on a server and then expose it to the world via a REST API so that my mobile app can access it. There are three components to this system: 

- **The SageMaker model instance and endpoint**: an instance of the trained model, with an endpoint allowing access to it.
- **A Lambda function**: AWS Lambda acts as a switchboard, using a small piece of your own code to pass images from the API to the model for prediction, and then to pass the model results to the API.
- **The API**: AWS API Gateway provides a REST API service.

So how does this all work? In my case it’s very simple: the API receives a POST request containing image data for prediction. The API passes the image data to the Lambda function, which converts it to the correct format and passes it to the SageMaker model instance via the model’s endpoint. The model makes a prediction and then returns a result to the Lambda, which passes it to the API and then out to the device that sent the image.

### Using the API

There's a notebook in my github project folder that shows how to use the microservice API I created ([dog_breed_id_test_API_manual.ipynb](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_test_API_manual.ipynb)) The notebook lets you select a JPG dog image on your local drive, converts the image to a base64 format, wraps it into a json payload and then passes it to the API URL via a POST request. The model prediction is returned in another json payload. For this notebook to work, you’d need to swap in your own deployed API URL, since I've shut down my own service.



### Testing model performance

Before you deploy a model to production, it's important to take a closer look at what the model is doing. How uniform is the model accuracy? Are there classes with poor performance, and if so, why and how can this be fixed? 

I [analyzed the trained model performance in more detail](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_test_API_valset.ipynb) by evaluating performance of a the test image set across classes. The model has 83% validation accuracy overall, but accuracy was not uniform: some classes performed better and others much worse.

![Dog mosaic](https://github.com/johnmburt/johnmburt.github.io/blob/master/images/projects/dog_id_validation_recall_per_breed.png)

The worst performing dog breed class was "Eskimo dog", with a recall score of just 3%. So, which other classes are "Eskimo dog" images being confused with? If I look at a confusion matrix, it appears that only two other classes are accounting for most of the errors: "malamute" and "Siberian husky".

![Dog mosaic](https://github.com/johnmburt/johnmburt.github.io/blob/master/images/projects/dog_id_validation_confusions.png)

Looking at some comparison images from each class is helpful way to track down the cause of the error: most of the images labeled "Eskimo dog" are actually huskies and malamutes: this is a labeling error. 

![Dog mosaic](https://github.com/johnmburt/johnmburt.github.io/blob/master/images/projects/dog_id_validation_confusions_examples.png)

Other low performing classes had different issues: the same breed split into multiple size classes (Poodle), or different breeds that just look very similar (Lhasa Apso, Maltese Dog, Shi Tzu). Another problem I found was that the images aren't well curated: there are pictures of plush toys and dogs obscured by objects and barely visible. A serious effort to build a production quality dog breed predictor would require a cleaner dataset with more accurate labeling. Perhaps this would be a good use of [Amazon SageMaker Ground Truth](https://aws.amazon.com/sagemaker/groundtruth/).


## Results summary:

Training the model was a fairly simple procedure, though sometimes it was a bit fussy to get everything configured just right. The classifier performed very well overall, with most classes at > 80% recall. Hyperparameter tuning was more complicated, and was expensive, since every training run cost money to complete. I would not recommend hyperparameter tuning except as an exercise, your model is performing badly or you're planning to deploy the model in production. For manual tuning, it helped to tune with a simpler task (in this case, classify only 10 breeds rather than the full 120), which greatly reduced training time.

Setting up the inference microservice was also reasonably simple, though again, there were some details that took me a while to figure out. However, once you know what to do, it is a suprisingly simple procedure. 

## Project notebooks:

#### [Generate LST files for SageMaker model training and validation](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_test_API_manual.ipynb)
- Generates the LST files necessary to train and test the model using SageMaker explorer. LST files describe the samples to use for training and testing the model. These files are uploaded to the S3 bucket folder that contains the train/test images and are used in the training job setup. This script allows you to specify a subset of classes to train (for model tuning), or all classes (final model training).

#### [Get sample info on a generated LST file set](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_classifier_get_LST_info.ipynb)
- This notebook shows how to read a LST file and print details about the classes. It can be used to get info for setting up hyperparams for a Sagemaker training job. Particularly the model definition will want to know number of classes and max number of samples in the training set.

#### [The Lambda function](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_lambda_function.ipynb)
- This is the function used by the Lambda service to pass images received via the API to the model endpoint for inference. You would paste this script into the Lambda during setup. This code was modified from the one presented in class to allow batches of images to be passed in one API call. 

#### [Test the API by sending a dog image](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_test_API_manual.ipynb)
- Manually test the trained model instance, via an AWS Gateway API call. This notebook lets you select a local image file, prepares the image data, passes it to the model for inference, then displays results.

#### [Analyze classifier performance via the API](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_test_API_valset.ipynb)
- Analyze the trained model performance in more detail by evaluating performance of a the test image set across classes. The model has 83% validation accuracy overall, but is that uniform, or are there some classes that perform better and others worse? I did find several "problem" classes in this analysis.


# AWS SageMaker training and microservice API setup details

## The training job

- Create an [IAM role permission](https://docs.aws.amazon.com/glue/latest/dg/create-an-iam-role-sagemaker-notebook.html). Once created, this can be used any time an IAM role is required.
- From the SageMaker console, select “Create training job”.
- Set the IAM role to the one you created above.
- Enter a unique job name (note: leave any settings not mentioned as defaults).
- Select “Amazon SageMaker built-in algorithm”, choose “Image classification”
- Select instance type ml.p2.xlarge, with instance count 2, set additional storage volume to 70 GB.
- Set hyperparameters as above.
- Create four input data channel configurations:
	- Channel name: train, Content type: application/x-image, Data source: S3, S3 location: s3://MY_BUCKET_NAME/dogs
	- Channel name: validation, Content type: application/x-image, Data source: S3, S3 location: s3://MY_BUCKET_NAME/dogs
	- Channel name: train_lst, Content type: application/x-image, Data source: S3, S3 location: s3://MY_BUCKET_NAME/dogs/dog_breeds_all_fold_1_train.lst
	- Channel name: validation_lst, Content type: application/x-image, Data source: S3, S3 location: s3://MY_BUCKET_NAME/dogs/dog_breeds_all_fold_1_test.lst
- Set S3 output path to “s3://MY_BUCKET_NAME/dogs/output”. This is where SageMaker will save the trained model artifact.

## Creating the inference microservice

## Setting up the model endpoint
- From SageMaker Studio, in Training jobs, select your finished training batch.
- Under Options, select “Create model”. Enter a model name, and choose the IAM role created earlier. Leave everything else alone and click “Create”. If all goes well, you will be in the Models section. 
- Select your model and choose “Create endpoint”. Give your endpoint a name. In the Endpoint configuration section select “Create a new endpoint configuration” and enter a configuration name, click “Create”. You should now be able to create the endpoint by clicking “Create” (again). It will take a little while for this endpoint to spin up – when it’s done, the status will change to “InService”.

## Giving Lambda permission to access the SageMaker model.
- Before creating the Lambda function, you need to set up a new custom IAM role to give Lambda permission to access the model endpoint.
- For that, go to the IAM console.
- With AWS service as the trusted entity, choose Lambda, then click “Next: Permissions”.
- Enter “SageMaker” in the search bar and select “AmazonSageMakerFullAccess”.
- Click “Next: Tags”.
- Click “Next: Review”. Give the new role a name, click “Create”.
- Now you should have a new role

## Setting up the Lambda function
- Navigate to [AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html), where you’ll set up a Lambda function to pass data between the model endpoint and the REST API server.
- In Functions, click “Create function”, select “Author from scratch”, give your lambda function a name, for Runtime select Python 3.8.
- Under “Choose or create an execution role”, click “Use an existing role”, then select the IAM role you created above. Then click “Create function”.
- You should have a new Lambda function.
- Now, you need to add the Python code to handle API POST input containing images to pass to the classifier for prediction. [I’ve posted my Python code here](https://github.com/johnmburt/projects/blob/master/AWS/sagemaker_dog_breed_id/dog_breed_id_lambda_function.ipynb). Grab this code and paste it into the Lambda code editor.
You’ll also need to add an environment variable called ENDPOINT_NAME and set it to the name of your model endpoint. 
- Click “Save” and now your Lambda is ready to hook up to an API.

## Setting up the REST API
- Navigate to [AWS API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html) to create and configure a REST API for your model.
- Click “Create API”, find “REST API” and click “Build”.
- Enter an API name and click “Create API”.
- Once the API is created you’ll be in the Resources section. From there, click the Actions tab and select “Create resource”. Name the resource “predict”, click “Create resource”.
-  Select the resource, then click the Actions tab again and now select “Create method”. In the dropdown, select “POST”, click the checkmark to create it.
- In the setup screen, enter the name of your lambda function, click “Save”.
- Go back to the Actions tab and click “Deploy API”, enter a stage name like “beta” and click “Deploy”.
- Now, you’ve deployed your API! You can get the URL if you open the stages tree and click on the POST method.

