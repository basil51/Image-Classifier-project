# Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. In this project, I  first develop code for an image classifier built with TensorFlow
you can review the file:
(Project_Image_Classifier_Project.ipynb) - which uses jupyter format and the file
Project_Image_Classifier_Project.py - same content but full python code

both upper files after runing will create the model file:
flower_classifier_model.keras

also you can create the same file model in a faster way by runing pythonn file:
train.py

Then I  convert it into a command line application.
I divided the command line app to two parts 
predict.py
and utils.py

the main command line is predict which import three functions from utils.py

In order to complete this project, I should use GPU but when I start the project the site tell me that I am out of GPU time , I sent and ask the udacity support and the told me they will fix it but I finshed the project on my CPU and I hope that my work is correct.

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.
