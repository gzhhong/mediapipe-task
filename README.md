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
The new model can't work in the iphone app.

The error message is:

```
HandLandmarker(15276,0x1ecb70c00) malloc: xzm: failed to initialize deferred reclamation buffer (46)
!!!!!!! HandLandmarkerService initialized !!!!!!!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1739095313.881470 7457008 gl_context.cc:357] GL version: 3.0 (OpenGL ES 3.0 Metal - 101), renderer: Apple A15 GPU
Initialized TensorFlow Lite runtime.
INFO: Initialized TensorFlow Lite runtime.
Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1739095313.894866 7457332 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1739095313.908151 7457333 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
!!!!!!! HandLandmarker created successfully !!!!!!!
F0000 00:00:1739095320.479028 7457331 tensors_to_landmarks_calculator.cc:145] Check failed: num_dimensions > 0 (0 vs. 0) 
*** Check failure stack trace: ***
    @        0x108dff4c8  absl::log_internal::LogMessage::SendToLog()
    @        0x108dfef10  absl::log_internal::LogMessage::Flush()
    @        0x108dff7f4  absl::log_internal::LogMessageFatal::~LogMessageFatal()
    @        0x108dff81c  absl::log_internal::LogMessageFatal::~LogMessageFatal()
    @        0x1090d8954  mediapipe::api2::TensorsToLandmarksCalculator::Process()
    @        0x10859daf8  mediapipe::api2::TensorsToLandmarksCalculator::Process()
    @        0x10870598c  mediapipe::CalculatorNode::ProcessNode()
    @        0x1086e42a0  mediapipe::internal::SchedulerQueue::RunCalculatorNode()
    @        0x1086e3d30  mediapipe::internal::SchedulerQueue::RunNextTask()
    @        0x1086fd944  mediapipe::ThreadPool::RunWorker()
    @        0x1086fd324  mediapipe::ThreadPool::WorkerThread::ThreadBody()
    @        0x2112c937c  _pthread_start
    @        0x2112c4494  thread_start
```