[[Japanese](README.md)/English]

> **Note**
> <br>I created a repository of model zoo for keypoint classification.
> <br>→ [Kazuhito00/hand-keypoint-classification-model-zoo](https://github.com/Kazuhito00/hand-keypoint-classification-model-zoo)

# hand-gesture-recognition-using-mediapipe
This is a simple program that estimates hand pose using MediaPipe (Python version), and recognizes hand signs and finger gestures using a simple MLP with the detected keypoints.
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

This repository includes:
* Sample program
* Hand sign recognition model (TFLite)
* Finger gesture recognition model (TFLite)
* Training data and notebook for hand sign recognition
* Training data and notebook for finger gesture recognition

# Requirements
* mediapipe 0.8.4
* OpenCV 4.6.0.66 or Later
* Tensorflow 2.9.0 or Later
* protobuf <3.20,>=3.9.2
* scikit-learn 1.0.2 or Later (only if you want to display the confusion matrix during training)
* matplotlib 3.5.1 or Later (only if you want to display the confusion matrix during training)

# Demo
Here's how to run the demo using a webcam:
```bash
python app.py
```

Here's how to run the demo using Docker and a webcam:
```bash
docker build -t hand_gesture .

xhost +local: && \
docker run --rm -it \
--device /dev/video0:/dev/video0 \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
-e DISPLAY=$DISPLAY \
hand_gesture:latest

python app.py
```

The following options can be specified when running the demo:
* --device<br>Specify the camera device number (Default: 0)
* --width<br>Width when capturing camera (Default: 960)
* --height<br>Height when capturing camera (Default: 540)
* --use_static_image_mode<br>Whether to use static_image_mode for MediaPipe inference (Default: not specified)
* --min_detection_confidence<br>Threshold for detection confidence (Default: 0.5)
* --min_tracking_confidence<br>Threshold for tracking confidence (Default: 0.5)

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is the sample program for inference. <br>You can also collect training data (keypoints) for hand sign recognition and training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is the training script for the hand sign recognition model.

### point_history_classification.ipynb
This is the training script for the finger gesture recognition model.

### model/keypoint_classifier
This directory contains files related to hand sign recognition. <br>The following files are included:
* Training data (keypoint.csv)
* Trained model (keypoint_classifier.tflite)
* Label data (keypoint_classifier_label.csv)
* Inference class (keypoint_classifier.py)

### model/point_history_classifier
This directory contains files related to finger gesture recognition. <br>The following files are included:
* Training data (point_history.csv)
* Trained model (point_history_classifier.tflite)
* Label data (point_history_classifier_label.csv)
* Inference class (point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
You can add or modify training data and retrain the models for hand sign recognition and finger gesture recognition.

### Hand sign recognition training method
#### 1. Collect training data
Press "k" to enter the mode for saving keypoints ("MODE:Logging Key Point" will be displayed)<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
Press "0" to "9" to append keypoints to "model/keypoint_classifier/keypoint.csv" as follows:<br>
Column 1: pressed number (used as class ID), Column 2 and beyond: keypoint coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The keypoint coordinates saved are those after the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
Initially, there are training data for three classes: paper (class ID: 0), rock (class ID: 1), and pointing (class ID: 2).<br>
Add data for class IDs 3 and beyond or delete existing data in the csv file as needed to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2. Train the model
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and run it from top to bottom.<br>
If you change the number of classes in the training data, change the value of "NUM_CLASSES = 3" and modify the labels in "model/keypoint_classifier/keypoint_classifier_label.csv" accordingly.<br><br>

#### X. Model structure
The model used in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" has the following structure:
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Finger gesture recognition training method
#### 1. Collect training data
Press "h" to enter the mode for saving the fingertip coordinate history ("MODE:Logging Point History" will be displayed)<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
Press "0" to "9" to append the coordinate history to "model/point_history_classifier/point_history.csv" as follows:<br>
Column 1: pressed number (used as class ID), Column 2 and beyond: coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The coordinate history saved is after the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
Initially, there are training data for four classes: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and movement (class ID: 4).<br>
Add data for class IDs 5 and beyond or delete existing data in the csv file as needed to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2. Train the model
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and run it from top to bottom.<br>
If you change the number of classes in the training data, change the value of "NUM_CLASSES = 4" and modify the labels in "model/point_history_classifier/point_history_classifier_label.csv" accordingly.<br><br>

#### X. Model structure
The model used in "[point_history_classification.ipynb](point_history_classification.ipynb)" has the following structure:
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. To use it, change "use_lstm = False" to "True" (requires tf-nightly as of 2020/12/16):<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Application example
Here are some application examples:
* [Control DJI Tello drone with Hand gestures](https://towardsdatascience.com/control-dji-tello-drone-with-hand-gestures-b76bd1d4644f)
* [Classifying American Sign Language Alphabets on the OAK-D](https://www.cortic.ca/post/classifying-american-sign-language-alphabets-on-the-oak-d)

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)

# Author
Kazuhito Takahashi (https://twitter.com/KzhtTkhs)

# License
hand-gesture-recognition-using-mediapipe is under the [Apache v2 license](LICENSE).