# Traffic Event Detection using Twitter Data
This repository contains the the implementation of machine learning based traffic inident detection using multi-modal data (Twitter data + Sensor data).

### System architecture

<img src="https://raw.githubusercontent.com/SenayGe/traffic-event-detection-using-Twitter-data/master/system_architecture.png" >
## Available modules and scripts
- main.py - Train and run the event detector
- tweets_fetch.py - Fetches tweets from twitter using twitter api
- filter_tweets.py - tweets preprocessing
- models - Defines our deep learning models
- training_ML - to trian via ML models (decision tree, Random Forest, XGboosting...)
- train_detector - module for training the detector

## Installation and setup
### Running the traffic-event detector
To `run` the detector:
```
python src\main.py
```


