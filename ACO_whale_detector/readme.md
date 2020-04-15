# Exploring deep sea acoustic events
## Whale song detector: CNN estimator model

### Feb 2020 PDSG Applied Data Science Meetup series<br>John Burt

Session details

For February’s four session meetup series we’ll be working with long term hydrophone recordings from University of Hawaii's Aloha Cabled Observatory (ACO - http://aco-ssds.soest.hawaii.edu), located at a depth of 4728m off Oahu. The recordings span a year and contain many acoustic events: wave movements, the sound of rain, ship noise, possible bomb noises, geologic activity and whale calls and songs. There is a wide range of project topics to explore: identifying and counting acoustic events such as whale calls, measuring daily or seasonal noise trends, measuring wave hydrodynamics, etc.
This notebook:

I built a classifier model to detect whale vocalizations in the recording. For this I used a standard Tensorflow/Keras Convolutional Neural Network (CNN), with a sound spectrograph as input.

The CNN model was trained using a generator function that combined background sounds from the hydrophone recording with whale vocalization clips selected from clean song recordings acquired from the Woods Hole Oceanographic Institution's Watkins Marine Mammal Sound Database. The generator function randomly combined noise and whale sounds so that each sample was unique.

