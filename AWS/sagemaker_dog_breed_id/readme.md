# Dog breed identification from images
## Training and deploying a SageMaker image-classifyer model as a microservice

### Class project for AICamp Full Stack Deep Learning in AWS

### John Burt
#### March - April 2020

For this project, the task was to create and train an image classifier using Amazon SageMaker, then deploy a classifier instance as a microservice via the AWS API Gateway. I chose to make a dog breed classifier using a set of images made available by Stanford University. 

### TLDR:

- Training the model was a fairly simple procedure, though a bit fussy. Hyperparameter tuning was more complicated, and was costly. 

- The classifier performed very well overall, with most classes at > 80% recall.

- Low performing classes appeared to have breed labelling errors (Eskimo Dog), were the result of the same breed split into multiple size classes (Poodle) or were different breeds that look very similar (Lhasa Apso, Maltese Dog, Shi Tzu).

- To improve model performance, I would:
  - Curate the training images, making sure breed ID was accurate and images were actually dogs (some were stuffed animals).
  - Combine breed size categories into one class.
  - Consider adding a size estimate feature (say, wither height).
  - Use image segmentation or manual annotation to define the image region containing the dog, then pass those regions to the model for training..

### Dataset:
The [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. Contents of this dataset (120 categories, 20580 images).

### The model

The model is a standard SageMaker image-classifier, which is a ResNet deep learning model. After tuning and testing, I chose the following hyperparameters:

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

### Other training details:
- The Model was trained on dog breed images uploaded to an S3 bucket. 
- Specified additional volume size of 70 GB.
- ml.p2.xlarge instance type, with 2 instances.
- I used LST files to specify train and test (validation) images. 
- The model training used early stopping.
- Enabled Managed Spot Training to reduce cost.
- The training job took 2.7 hrs, and validation accuracy was 83%.
- The learning rate was a bit low, causing slower training, but I had problems earlier w/ poor training at higher learning rates. I think an ideal rate would probably be 0.0005.

### Microservice setup
- After the job trained, I created a model instance.
- From the model instance, an endpoint was created.
- A lambda was created to process POST data from the Gateway API. The Lambda used an environment variable to specify the model endpoint.
- The API created had one resource, 'classify'. A POST method was attached, with pointed to the Lambda. 
- The Lambda function was modified to allow posting batches of multiple images.

## Project notebooks:


#### [Generate LST files for SageMaker model training and validation](dog_breed_id_test_API_manual.ipynb)
- This notebook generates the LST files necessary to train and test the model using SageMaker explorer. LST files describe the samples to use for training and testing the model. These files are uploaded to the S3 bucket folder that contains the train/test images and are used in the training job setup.


#### [Get sample info on a generated LST file set](dog_breed_classifier_get_LST_info.ipynb)
- This notebook can be used to help set up hyperparams for a Sagemaker training job. The model will want to know number of clases, and (max) number of samples (120 categories, 20580 images).


#### [The Lambda function](dog_breed_id_lambda_function.ipynb)
- This is the lambda function used by the microservice to pass received images via API posts to the model endpoint for inference. This Lambda was modified from the one presented in class to allow batches of images to be passed in one post. 

#### [Test the API by sending a dog image](dog_breed_id_test_API_manual.ipynb)
- Test the trained model instance, via an AWS Gateway API call. This notebook lets you select a local image file, prepares the image data, passes it to the model for inference, then displayes results.


#### [Analyze classifier performance via the API](dog_breed_id_test_API_valset.ipynb)
- This notebook examines the trained model performance in more detail. The model has 83% validation accuracy overall, but is that uniform, or are there some classes that perform better and other worse? 










