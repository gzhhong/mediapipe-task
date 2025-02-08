from tflite_support import metadata
import tensorflow as tf
import json

original_model_path = "./original_task/hand_landmarks_detector.tflite"
model_path = "./hand_landmark_model_float16.tflite"

# Check metadata
try:
    # Check and print metadata using the metadata display
    model_metadata = metadata.MetadataDisplayer.with_model_file(model_path)
    print(model_metadata.get_metadata_json())

    original_meta = metadata.MetadataDisplayer.with_model_file(original_model_path)
    print(original_meta.get_metadata_json())

    # compare the metadata  
    if model_metadata.get_metadata_json() == original_meta.get_metadata_json():
        print("Metadata successfully verified.")
    else:
        # highlight the difference
        model_meta_dict = json.loads(model_metadata.get_metadata_json())
        original_meta_dict = json.loads(original_meta.get_metadata_json())
        
        def compare_dict(dict1, dict2, path=""):
            differences = []
            for key in set(dict1.keys()) | set(dict2.keys()):
                current_path = f"{path}.{key}" if path else key
                if key not in dict1:
                    differences.append(f"Missing in model 1: {current_path}")
                elif key not in dict2:
                    differences.append(f"Missing in model 2: {current_path}")
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        differences.extend(compare_dict(dict1[key], dict2[key], current_path))
                    else:
                        differences.append(f"Different value at {current_path}:")
                        differences.append(f"  Model 1: {dict1[key]}")
                        differences.append(f"  Model 2: {dict2[key]}")
            return differences
        
        differences = compare_dict(model_meta_dict, original_meta_dict)
        print("Metadata differences found:")
        for diff in differences:
            print(diff)
        
        print("Metadata verification failed.")
except Exception as e:
    print("Error reading metadata:", str(e))
