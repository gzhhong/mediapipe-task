import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from config import IMAGE_EDGE

from train_mediapipe import (
    custom_loss, 
    custom_metric, 
    CoordinatesMergeLayer, 
    NegativeActivation,
    ScaleShiftLayer,
    parse_tfrecord,
    process_image_and_coords
)
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import flatbuffers

def test_tflite_model(model_path, test_image_path):
    """Test TFLite model"""
    print(f"Testing model: {model_path}")
    
    # get model type
    model_type = Path(model_path).stem.split('_')[-1]  # get 'float16'/'dynamic_range'/'int8'
    
    # load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nmodel input details:")
    print(f"input type: {input_details[0]['dtype']}")
    print(f"input quantization parameters: {input_details[0]['quantization']}")
    print(f"input shape: {input_details[0]['shape']}")
    
    print("\nmodel output details:")
    print(f"output type: {output_details[0]['dtype']}")
    print(f"output quantization parameters: {output_details[0]['quantization']}")
    print(f"output shape: {output_details[0]['shape']}")
    
    # load and preprocess test image
    img = cv2.imread(str(test_image_path))
    if img is None:
        raise ValueError(f"failed to read image: {test_image_path}")
    
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    path = Path(test_image_path)
    directory = path.parent
    filename = path.name

    # Replace 'frame' with 'landmarks' in the filename only
    new_filename = filename.replace('frame', 'landmarks', 1)
    new_filename = new_filename.replace('jpg', 'txt')

    # Reassemble the full path
    landmarks_file = directory / new_filename
    with open(landmarks_file, 'r') as f:
        lines = f.readlines()
        original_landmarks = []
        for line in lines:
            _, x, y, z = map(float, line.strip().split(','))  # use comma as separator
            original_landmarks.append([x, y, z])
    # use the same preprocessing as during training
    processed_img, original_landmarks = process_image_and_coords(img, original_landmarks, target_size=(IMAGE_EDGE, IMAGE_EDGE))
    
    # add batch dimension
    processed_img = np.expand_dims(processed_img, axis=0)
    
    print(f"\nprocessed input data type: {processed_img.dtype}")
    print(f"input data range: [{np.min(processed_img)}, {np.max(processed_img)}]")
    
    # set input data and run inference
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    
    # get output and reshape to (21, 3)
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    # print landmarks
    print(f"predict-landmarks: {landmarks}")
    landmarks = landmarks.reshape(-1, 21, 3)
    
    # print landmarks and original_landmarks and their mae
    original_landmarks = np.array(original_landmarks).reshape(1, 21, 3)
    mae = np.mean(np.abs(landmarks - original_landmarks), axis=0)
    print(f"landmarks: {landmarks}")
    print(f"original_landmarks: {original_landmarks}")
    print(f"mae: {mae}")
    # visualize result...
    img_display = cv2.imread(str(test_image_path))
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    processed_img = (processed_img[0] * 255).astype(np.uint8)  # Remove batch dimension and convert to uint8

    # draw keypoints and connecting lines
    for i, (x, y, z) in enumerate(landmarks[0]):
        height, width, _ = img_display.shape
        x_px = int(x )
        y_px = int(y )
        # adjust color based on z value (smaller z, darker color)
        color = (0, int(255 * (1 + z)), 0)
        # draw keypoint
        cv2.circle(processed_img, (x_px, y_px), 3, color, -1)
        # add label
        cv2.putText(processed_img, f"{i}", (x_px+4, y_px+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # save different result files for each model type
    output_image = f'tflite_test_result_{model_type}.jpg'
    cv2.imwrite(output_image, processed_img)
    print(f"\ntest result saved to: {output_image}")
    
    # save predicted coordinates to different text files
    output_txt = f'tflite_predictions_{model_type}.txt'
    with open(output_txt, 'w') as f:
        for i, (x, y, z) in enumerate(landmarks[0]):
            f.write(f"{i},{x:.6f},{y:.6f},{z:.6f}\n")
    print(f"predicted coordinates saved to: {output_txt}")
    
    return landmarks[0]

def create_metadata():
    """create metadata compliant with MediaPipe"""
    metadata_model = _metadata_fb.ModelMetadataT()
    metadata_model.name = "HandLandmarkDetector"
    metadata_model.description = "Identify the predefined landmarks on a potentially present hand andprovide information about their positions within the given image or avideo stream."
    metadata_model.version = "1"
    metadata_model.author = "MediaPipe"
    
    # create SubGraphMetadata
    subgraph = _metadata_fb.SubGraphMetadataT()
    
    # input tensor metadata
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = "Input image to be detected."
    
    # set input image properties
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties
    input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
    
    # set normalization parameters
    process_unit = _metadata_fb.ProcessUnitT()
    process_unit.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
    process_unit.options = _metadata_fb.NormalizationOptionsT()
    process_unit.options.mean = [0.0, 0.0, 0.0]
    process_unit.options.std = [255.0, 255.0, 255.0]
    input_meta.processUnits = [process_unit]
    
    # set input statistics
    input_meta.stats = _metadata_fb.StatsT()
    input_meta.stats.max = [1.0, 1.0, 1.0]
    input_meta.stats.min = [0.0, 0.0, 0.0]
    
    # create output tensor metadata list
    output_metas = []
    
    # 1. handedness output
    handedness_meta = _metadata_fb.TensorMetadataT()
    handedness_meta.name = "handedness"
    handedness_meta.description = "The score of left or right handedness."
    handedness_meta.content = _metadata_fb.ContentT()
    handedness_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    handedness_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
    handedness_meta.stats = _metadata_fb.StatsT()
    # add label file association
    handedness_meta.associatedFiles = [_metadata_fb.AssociatedFileT()]
    handedness_meta.associatedFiles[0].name = "handedness.txt"
    handedness_meta.associatedFiles[0].description = "Labels for categories that the model can recognize."
    handedness_meta.associatedFiles[0].type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    
    # 2. presence score output
    presence_meta = _metadata_fb.TensorMetadataT()
    presence_meta.name = "presense score"
    presence_meta.description = "The score of hand presence in the image."
    presence_meta.content = _metadata_fb.ContentT()
    presence_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    presence_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
    presence_meta.stats = _metadata_fb.StatsT()
    
    # 3. landmarks output
    landmarks_meta = _metadata_fb.TensorMetadataT()
    landmarks_meta.name = "landmarks"
    landmarks_meta.description = "The hand landmarks in normalized coordinates."
    landmarks_meta.content = _metadata_fb.ContentT()
    landmarks_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    landmarks_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
    landmarks_meta.stats = _metadata_fb.StatsT()
    
    # 4. world landmarks output
    world_landmarks_meta = _metadata_fb.TensorMetadataT()
    world_landmarks_meta.name = "world landmarks"
    world_landmarks_meta.description = "The hand landmarks in world coordinates."
    world_landmarks_meta.content = _metadata_fb.ContentT()
    world_landmarks_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
    world_landmarks_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.FeatureProperties
    world_landmarks_meta.stats = _metadata_fb.StatsT()
    
    # add all output metadata to list
    output_metas.extend([landmarks_meta, presence_meta, handedness_meta, world_landmarks_meta])
    
    # add to subgraph
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = output_metas
    metadata_model.subgraphMetadata = [subgraph]
    
    # create metadata buffer
    builder = flatbuffers.Builder(0)
    metadata_buf = metadata_model.Pack(builder)
    builder.Finish(metadata_buf, _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    return bytes(builder.Output())

def add_metadata_to_model(model_buffer):
    """add metadata to model"""
    try:
        # create handedness.txt file content
        handedness_labels = ["Left", "Right"]
        handedness_txt = "\n".join(handedness_labels)
        
        # create actual handedness.txt file
        handedness_file_path = Path("handedness.txt")
        with open(handedness_file_path, "w") as f:
            f.write(handedness_txt)
        
        # create metadata
        metadata_buf = create_metadata()
        model_buffer_bytes = bytes(model_buffer)
        creator = _metadata.MetadataPopulator.with_model_buffer(model_buffer_bytes)
        
        # load metadata buffer
        creator.load_metadata_buffer(metadata_buf)
        
        # load handedness label file
        creator.load_associated_files({"handedness.txt": str(handedness_file_path)})
        
        # populate metadata
        creator.populate()
        
        # delete temporary file
        handedness_file_path.unlink()
        
        return creator.get_model_buffer()
    except Exception as e:
        print(f"error adding metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        # ensure cleanup of temporary file
        if 'handedness_file_path' in locals() and handedness_file_path.exists():
            handedness_file_path.unlink()
        return model_buffer  # if adding metadata fails, return original model

def convert_to_tflite_float16(model_path='./hand_landmark_model.keras', output_path='./hand_landmark_model_float16.tflite'):
    """convert to float16 TFLite model"""
    # load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'custom_loss': custom_loss,
            'custom_metric': custom_metric,
            'CoordinatesMergeLayer': CoordinatesMergeLayer,
            'NegativeActivation': NegativeActivation,
            'ScaleShiftLayer': ScaleShiftLayer
        }
    )
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # set float16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    try:
        print("converting to float16 model...")
        tflite_model = converter.convert()
        
        # add metadata
        print("adding metadata...")
        tflite_model_with_metadata = add_metadata_to_model(tflite_model)
        if tflite_model_with_metadata is None:
            print("adding metadata failed, saving original model...")
            tflite_model_with_metadata = tflite_model
        
        print("conversion complete, saving model...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model_with_metadata)
        print(f"float16 model saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"float16 conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_tflite_float32(model_path='./hand_landmark_model.keras', output_path='./hand_landmark_model_float32.tflite'):
    """Convert to TFLite model with float32 precision"""
    # Load the model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'custom_loss': custom_loss,
            'custom_metric': custom_metric,
            'CoordinatesMergeLayer': CoordinatesMergeLayer,
            'NegativeActivation': NegativeActivation,
            'ScaleShiftLayer': ScaleShiftLayer
        }
    )

    # Convert to TFLite without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # No optimization, no quantization
    converter.experimental_new_converter = True

    try:
        print("Converting to float32 TFLite model...")
        tflite_model = converter.convert()

        # Add metadata (optional)
        print("Adding metadata...")
        tflite_model_with_metadata = add_metadata_to_model(tflite_model)
        if tflite_model_with_metadata is None:
            print("Metadata addition failed, saving raw model...")
            tflite_model_with_metadata = tflite_model

        print("Conversion complete, saving model...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model_with_metadata)
        print(f"Float32 model saved to: {output_path}")

        return True
    except Exception as e:
        print(f"Float32 conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def convert_to_tflite_dynamic(model_path='./hand_landmark_model.keras', output_path='./hand_landmark_model_dynamic_range.tflite'):
    """convert to dynamic range quantized TFLite model"""
    # load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'custom_loss': custom_loss,
            'custom_metric': custom_metric,
            'CoordinatesMergeLayer': CoordinatesMergeLayer,
            'NegativeActivation': NegativeActivation,
            'ScaleShiftLayer': ScaleShiftLayer
        }
    )
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # set dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        print("converting to dynamic range quantized model...")
        tflite_model = converter.convert()
        
        # add metadata
        print("adding metadata...")
        tflite_model_with_metadata = add_metadata_to_model(tflite_model)
        if tflite_model_with_metadata is None:
            print("adding metadata failed, saving original model...")
            tflite_model_with_metadata = tflite_model
        
        print("conversion complete, saving model...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model_with_metadata)
        print(f"dynamic range quantized model saved to: {output_path}")
        
        return True
    except Exception as e:
        print(f"dynamic range quantization conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # convert to float16 model
    print("\n1. converting to float16 model")
    float16_success = convert_to_tflite_float16()
    
    # convert to dynamic range quantized model
    print("\n2. converting to dynamic range quantized model")
    dynamic_success = convert_to_tflite_dynamic()
    
    # convert to float32 model
    print("\n3. converting to float32 model")
    float32_success = convert_to_tflite_float32()
    
    # print summary
    print("\nconversion result summary:")
    print(f"float16 model: {'success' if float16_success else 'failed'}")
    print(f"dynamic range quantized model: {'success' if dynamic_success else 'failed'}")
    print(f"float32 model: {'success' if float32_success else 'failed'}")
    
    # if all conversions are successful, test the models
    if float16_success and dynamic_success and float32_success:
        test_image_path = './frame_20250121_103737_986144.jpg'
        if Path(test_image_path).exists():
            print("\ntesting all converted models:")
            print("\n1. testing float16 model")
            test_tflite_model('hand_landmark_model_float16.tflite', test_image_path)
            print("\n2. testing dynamic range quantized model")
            test_tflite_model('hand_landmark_model_dynamic_range.tflite', test_image_path)
            print("\n3. testing float32 model")
            test_tflite_model('hand_landmark_model_float32.tflite', test_image_path)
        else:
            print("test image not found")
    else:
        print("\nskipping testing phase due to model conversion failure")

if __name__ == "__main__":
    main() 