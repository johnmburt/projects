# Exploring deep sea acoustic events
## Whale song detector: CNN estimator model

### Feb 2020 PDSG Applied Data Science Meetup series<br>John Burt

#### Session details

For February’s four session meetup series we worked with long term hydrophone recordings from University of Hawaii's Aloha Cabled Observatory (ACO - http://aco-ssds.soest.hawaii.edu), located at a depth of 4728m off Oahu. The recordings span a year and contain many acoustic events: wave movements, the sound of rain, ship noise, possible bomb noises, geologic activity and whale calls and songs. There is a wide range of project topics to explore: identifying and counting acoustic events such as whale calls, measuring daily or seasonal noise trends, measuring wave hydrodynamics, etc.

For my analysis I chose to develop a humpback whale song detector. Humpbacks spend their breeding season off the Hawaiian Islands and are very vocal during that time. For the project, I located and downloaded a library of humpback whale song and conducted EDA on the songs to inform my model design. Then I built a CNN based classifier model, trained it, did validation testing and then ran the model on a year's worth of hydrophone recordings. I then analyzed the resulting detection data and found some interesting trends in the frequency of singing throughout the day and across the year.

#### Notebooks in this project folder:

[EDA for whale example clips](ACO_whalesong_detector_target_sound_EDA_Vf.ipynb)
- For the whale song detector, I needed to prepare a set of clean song examples to train my classifier with. For this I clipped several hundred examples of humpback whale song from WHOI Watson library recordings that I downloaded. The clips are of individual song notes and range from ~ 1 sec to 8 sec long. I analyzed the details of these vocalizations, and found that there are three types of notes that whales produce, distinguishable by frequency range and other details.


[Pre-process target and background audio](ACO_whalesong_detector_preprocess_target_and_background_vf.ipynb)
- For model training and testing, I overlay whale song note example clips with recording background noise. In this notebook, I clean and prepare the training target sounds (whale song notes), and background sound, then save them in HDF5 files for quick loading during training runs.


[Whale song detector: CNN estimator model](ACO_whalesong_detector_CNN_clf_model_vf.ipynb)
- I built a classifier model to detect whale vocalizations in the recording. For this I used a standard Tensorflow/Keras Convolutional Neural Network (CNN), with a sound spectrograph as input. The CNN model was trained using a generator function that combined background sounds from the hydrophone recording with whale vocalization clips selected from clean song recordings acquired from the Woods Hole Oceanographic Institution's Watkins Marine Mammal Sound Database. The generator function randomly combined noise and whale sounds so that each sample was unique.


[Whale song detector scan results](ACO_whalesong_detector_model_results_vf.ipynb)
- Analysis of the results of my CNN classifier's detection scan of the "every_other_hour" recording. This recording spans all of 2015, with 5 minutes of audio sampled every other hour. The detection script saves the audio for any frames the model classifies as having whale song to a detections folder for later review. I then examined the results after manually scanning every detection and entering a comment if the clip does not contain whale song.

