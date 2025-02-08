## Installation
python3.9 and venv
pip install tensorflow keras six tflite_support

## original_task
Contains the original task file hand_landmarks_detector.tflite. We will use this file to check the metadata in the new tflite file created by convert_to_tflite.py.

## convert tflite
The hand_landmark_model_float16.tflite is converted from hand_landmark_model.keras by convert_to_tflite.py.

About how to generate the keras and h5 model, please refer to the training_mediapipe.py in project https://github.com/gzhhong/mediapipe

During the converting process, the metadata is also generated and written in the tflite file.

## check the metadata
check_meta_in_tflite.py is used to check the metadata in the new tflite file, hand_landmark_model_float16.tflite with the metadata in the original file ./original_task/hand_landmarks_detector.tflite.

The output shows currently the metadata in the two files are same.

## create task file
create_task_file.py is used to create the task file for the iphone app, it will remove the hand_landmarks_detector.tflite in the new_task folder and copy the hand_landmark_model_float16.tflite to the new_task folder and rename it to hand_landmarks_detector.tflite. Then create the task file by compressing the new_task folder.

## Issue
Normally, if the tflite file has the same metadata as the original tflite model in the old task, the new task file should be able to be recognized in the iphone app. But the new task file is not working, the model is not able to be recognized in iphone app. 

The error message is:

```
!!!!!!! HandLandmarkerService initialized !!!!!!!
!!!!!!! Failed to create HandLandmarker: Error Domain=com.google.mediapipe.tasks Code=3 "INVALID_ARGUMENT: Invalid metadata schema version: expected M001, got " UserInfo={NSLocalizedDescription=INVALID_ARGUMENT: Invalid metadata schema version: expected M001, got } !!!!!!!
INVALID_ARGUMENT: Invalid metadata schema version: expected M001, got 
```