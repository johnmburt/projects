# Music clip dataset for genre ID and other applications

The data for this project is from ["FMA: A Dataset For Music Analysis"](https://github.com/mdeff/fma). That dataset is a dump of sound clips and associated metadata from the Free Music Archive (FMA), an interactive library of high-quality, legal audio downloads. 


For the meetup series, the music data was reduced further to a "warmup set" of 4000 samples each of folk and hip-hop music. This code uses the warmup set, but is also tooled to use the original music clip dataset, which is much larger and has more genre categories.


## Projects:

### Audio feature visualization

Using the librosa audio analysis package, I visualized music clips. The goal was to find a set of features that contained enough information to identify what genre a song clip belonged to. I reasonsed that harmonic and tempo and beat information would be a good place to start.

- [music_genre_id/music_genre_id_visualize_features_vf.ipynb](http://nbviewer.jupyter.org/github/johnmburt/projects/blob/master/music_genre_id/music_genre_id_visualize_features_vf.ipynb) 

### Classifier feature generation

I generated feature data from from all of the sound clips and saved those features to a csv file. This extra step was necessary because feature generation can take a long time.

- [music_genre_id_feature_generation_vf.ipynb](http://nbviewer.jupyter.org/github/johnmburt/projects/blob/master/music_genre_id/music_genre_id_feature_generation_vf.ipynb) 


### Music genre classification

With the feature data I generated from the sound clips, I trained and tested an XGBoost music genre classifier. The classifier is straightforward, using default parameters, and gets pretty good accuracy (88% on the holdout test set). 

- [music_genre_id_classifier_xgboost_predict_test_vf.ipynb](http://nbviewer.jupyter.org/github/johnmburt/projects/blob/master/music_genre_id/music_genre_id_classifier_xgboost_predict_test_vf.ipynb) 
