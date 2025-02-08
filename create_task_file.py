import zipfile
import os
import shutil
# Path to the folder where the models were extracted
folder_path = "new_task"

# Path where the new .task file will be saved
new_task_file = "hand_landmarker_new.task"

# remove the hand_landmarks_detector.tflite in the new_task folder
os.remove("new_task/hand_landmarks_detector.tflite")
# copy the hand_landmark_model_float16.tflite to the new_task folder
shutil.copy("hand_landmark_model_float16.tflite", "new_task/hand_landmark_model_float16.tflite")
# rename the hand_landmark_model_float16.tflite to hand_landmarks_detector.tflite
os.rename("new_task/hand_landmark_model_float16.tflite", "new_task/hand_landmarks_detector.tflite")

# Recreate the .task file by compressing the folder without compression
with zipfile.ZipFile(new_task_file, 'w', zipfile.ZIP_STORED) as zip_ref:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            zip_ref.write(os.path.join(root, file),
                          os.path.relpath(os.path.join(root, file), folder_path))

print(f"Task file created: {new_task_file}")

