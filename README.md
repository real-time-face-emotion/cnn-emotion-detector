# Real-Time Facial Emotion Recognition Using Deep Learning

### Hey There! What’s This About?


Welcome to my cool project! This repo holds a real-time facial emotion recognition system that uses a Convolutional Neural Network (CNN) to figure out what you’re feeling—anger, disgust, fear, happiness, sadness, surprise, or neutral—right from your webcam. I built it with TensorFlow/Keras for the brainy model part and OpenCV to catch faces and show results live. It’s pretty neat, hitting 85% accuracy on the FER2013 dataset!

### What Do You Need to Get Started?

Before diving in, make sure you’ve got:



Python 3.7 or newer (the coding language we’re using)



pip (to grab the needed tools)

You’ll also need these Python libraries:\

tensorflow (2.x version)



keras

opencv-python

numpy

pandas

matplotlib

seaborn

imblearn

scikit-learn

### How to Use It

##### Train the Model

Fire up DL.ipynb in Jupyter Notebook or Google Colab.

Make sure fer2013.csv is in the folder.

Run all the cells to prep the data, train the CNN, and save the model as emotion_recognition_model.h5. It uses some smart tricks like data augmentation and oversampling to balance things out. A GPU (like Colab’s T4) speeds it up!

### See It Live


Check that emotion_recognition_model.h5 and haarcascade_frontalface_default.xml (included) are in the folder.

Run this to start the fun:

python real_time_emotion.py

A window pops up with your webcam feed, showing faces with emotion labels like “Happiness” in a green box.

Hit q to close it when you’re done.

### What to Expect





It runs at about 15 frames per second (FPS) on a regular CPU—pretty smooth!



You’ll see green boxes around faces with the guessed emotion on top.

### What’s in the Files?





DL.ipynb: The Jupyter Notebook where the magic happens—data prep, training, and cool plots like accuracy and confusion matrix.



real_time_emotion.py: The script that brings it to life with your webcam.



emotion_recognition_model.h5: The pre-trained model ready to roll (85% accurate!).



haarcascade_frontalface_default.xml: OpenCV’s face detector file.



README.md: This file—you’re reading it!



requirements.txt: (Make this!) A list of all the tools you need.

### About the Dataset

The FER2013 dataset isn’t here—it’s not my call to include it. Here’s the scoop:





Size: 35,887 grayscale images (48x48 pixels).



Emotions: Anger, disgust, fear, happiness, sadness, surprise, neutral.



Split: 28,709 for training, 3,589 for validation, 3,589 for testing.



Where to Get It: Kaggle—check the rules there!



Tip: Put fer2013.csv in the project folder to get started.

### Wanna Help Out?

Love the project? Jump in! Here’s how:





1.Fork this repo.



2.Make a new branch (git checkout -b cool-new-feature).



3.Tweak the code and commit (git commit -m "Added awesome stuff").



4.Push it (git push origin cool-new-feature).



5.Open a pull request!

Keep it clean—follow PEP 8 style and add comments so I can follow along.

### License

This is under the MIT License—feel free to use it, tweak it, whatever! Check the LICENSE file for the details. (Add a LICENSE file with MIT terms if you haven’t yet.)

Get in Touch

Got questions or ideas? Hit me up:





Last updated: April 19, 2025
