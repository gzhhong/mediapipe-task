import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train_mediapipe import (
    parse_tfrecord,
    custom_loss,
    CoordinatesMergeLayer,
    NegativeActivation
)

def prepare_test_data():
    """prepare test data"""
    # load a test sample
    tfrecord_path = str(Path(__file__).parent / 'training_data.tfrecord')
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    
    # get a sample
    for image, outputs in dataset.take(1):
        return image, outputs['landmarks']
    
    raise RuntimeError("failed to load test data")

def visualize_comparison(image, true_landmarks, pred_landmarks, title="Prediction vs Ground Truth"):
    """visualize prediction vs ground truth"""
    plt.figure(figsize=(15, 5))
    
    # ensure image is correct type and value range
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    # ensure landmarks are numpy array
    if isinstance(true_landmarks, tf.Tensor):
        true_landmarks = true_landmarks.numpy()
    if isinstance(pred_landmarks, tf.Tensor):
        pred_landmarks = pred_landmarks.numpy()
    
    # reshape landmarks to [21, 3] shape
    true_landmarks = true_landmarks.reshape(-1, 3)
    pred_landmarks = pred_landmarks.reshape(-1, 3)
    
    # display true and predicted points on original image
    plt.subplot(131)
    plt.imshow(image)
    
    # finger connection definition
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
    ]
    
    # draw true points connection (green)
    for start, end in connections:
        plt.plot([true_landmarks[start, 0] * image.shape[1], true_landmarks[end, 0] * image.shape[1]],
                [true_landmarks[start, 1] * image.shape[0], true_landmarks[end, 1] * image.shape[0]],
                'g-', alpha=0.5, label='True' if start == 0 else "")
    
    # draw predicted points connection (red)
    for start, end in connections:
        plt.plot([pred_landmarks[start, 0] * image.shape[1], pred_landmarks[end, 0] * image.shape[1]],
                [pred_landmarks[start, 1] * image.shape[0], pred_landmarks[end, 1] * image.shape[0]],
                'r-', alpha=0.5, label='Pred' if start == 0 else "")
    
    # draw points
    plt.scatter(true_landmarks[:, 0] * image.shape[1], true_landmarks[:, 1] * image.shape[0],
               c='g', s=30, label='True Points')
    plt.scatter(pred_landmarks[:, 0] * image.shape[1], pred_landmarks[:, 1] * image.shape[0],
               c='r', s=30, label='Pred Points')
    
    # add point labels
    for i, ((tx, ty, _), (px, py, _)) in enumerate(zip(true_landmarks, pred_landmarks)):
        plt.annotate(f'{i}', (tx * image.shape[1], ty * image.shape[0]),
                    color='g', fontsize=8)
        plt.annotate(f'{i}', (px * image.shape[1], py * image.shape[0]),
                    color='r', fontsize=8)
    
    plt.legend()
    plt.title(title)
    
    # display error distribution
    plt.subplot(132)
    errors = np.sqrt(np.sum((true_landmarks - pred_landmarks) ** 2, axis=1))
    plt.bar(range(21), errors)
    plt.title('Point-wise Error Distribution')
    plt.xlabel('Point Index')
    plt.ylabel('Euclidean Error')
    
    # display z coordinate comparison
    plt.subplot(133)
    plt.plot(true_landmarks[:, 2], label='True Z', color='g')
    plt.plot(pred_landmarks[:, 2], label='Pred Z', color='r')
    plt.title('Z Coordinate Comparison')
    plt.xlabel('Point Index')
    plt.ylabel('Z Value')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()

def test_tflite_model(model_path):
    """test TFLite model"""
    print(f"\ntesting TFLite model: {Path(model_path).name}")
    
    try:
        # load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # print input size
        input_shape = input_details[0]['shape']
        print(f"TFLite model input size: {input_shape[1]}x{input_shape[2]}")
        
        # prepare test data
        image, true_landmarks = prepare_test_data()
        
        # expand dimension to match model input
        input_data = tf.expand_dims(image, 0)

        # set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # run inference
        interpreter.invoke()
        
        # get output
        landmarks_output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"Input shape (TFLite): {input_details[0]['shape']}")
        print(f"Output shape (TFLite): {output_details[0]['shape']}")
        print(f"Landmarks output (TFLite): {landmarks_output}")
        # calculate error
        true_reshaped = tf.reshape(true_landmarks, [-1, 21, 3])
        pred_reshaped = tf.reshape(landmarks_output, [-1, 21, 3])
        
        # calculate MAE and MSE
        mae = tf.reduce_mean(tf.abs(true_reshaped - pred_reshaped))
        mse = tf.reduce_mean(tf.square(true_reshaped - pred_reshaped))
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # calculate xy and z error
        xy_mae = tf.reduce_mean(tf.abs(true_reshaped[..., :2] - pred_reshaped[..., :2]))
        xy_mse = tf.reduce_mean(tf.square(true_reshaped[..., :2] - pred_reshaped[..., :2]))
        z_mae = tf.reduce_mean(tf.abs(true_reshaped[..., 2] - pred_reshaped[..., 2]))
        z_mse = tf.reduce_mean(tf.square(true_reshaped[..., 2] - pred_reshaped[..., 2]))
        
        print("\ncoordinate component error:")
        print(f"XY - MSE: {xy_mse:.6f}, MAE: {xy_mae:.6f}")
        print(f"Z  - MSE: {z_mse:.6f}, MAE: {z_mae:.6f}")
        
        # calculate important points error
        important_points = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        true_important = tf.gather(true_reshaped, important_points, axis=1)
        pred_important = tf.gather(pred_reshaped, important_points, axis=1)
        
        important_mae = tf.reduce_mean(tf.abs(true_important - pred_important))
        important_mse = tf.reduce_mean(tf.square(true_important - pred_important))
        
        print("\nimportant points error:")
        print(f"MSE: {important_mse:.6f}")
        print(f"MAE: {important_mae:.6f}")
        
        # visualize result
        fig = visualize_comparison(image, true_landmarks, landmarks_output,
                                 title=f'TFLite Model\nMSE: {mse:.6f}, MAE: {mae:.6f}')
        fig.savefig(f'tflite_model_results.png')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_and_test_model(model_path):
    """load and test model"""
    print(f"\ntesting model: {Path(model_path).name}")
    
    try:
        # prepare custom objects dictionary
        custom_objects = {
            'custom_loss': custom_loss,
            'CoordinatesMergeLayer': CoordinatesMergeLayer,
            'NegativeActivation': NegativeActivation,
            'mse': 'mse'  # directly use string 'mse'
        }
        
        # load model
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("model loaded successfully")
        
        # prepare test data
        image, true_landmarks = prepare_test_data()
        
        # expand dimension to match model input
        input_data = tf.expand_dims(image, 0)
        
        # run prediction
        predictions = model.predict(input_data)
        
        # get landmarks prediction
        pred_landmarks = predictions['landmarks']
        print(f"Keras Output shape: {pred_landmarks.shape}")
        print(f"Keras Landmarks output: {pred_landmarks}")
        # calculate error
        true_reshaped = tf.reshape(true_landmarks, [-1, 21, 3])
        pred_reshaped = tf.reshape(pred_landmarks, [-1, 21, 3])
        
        # calculate MAE and MSE
        mae = tf.reduce_mean(tf.abs(true_reshaped - pred_reshaped))
        mse = tf.reduce_mean(tf.square(true_reshaped - pred_reshaped))
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # calculate xy and z error
        xy_mae = tf.reduce_mean(tf.abs(true_reshaped[..., :2] - pred_reshaped[..., :2]))
        xy_mse = tf.reduce_mean(tf.square(true_reshaped[..., :2] - pred_reshaped[..., :2]))
        z_mae = tf.reduce_mean(tf.abs(true_reshaped[..., 2] - pred_reshaped[..., 2]))
        z_mse = tf.reduce_mean(tf.square(true_reshaped[..., 2] - pred_reshaped[..., 2]))
        
        print("\ncoordinate component error:")
        print(f"XY - MSE: {xy_mse:.6f}, MAE: {xy_mae:.6f}")
        print(f"Z  - MSE: {z_mse:.6f}, MAE: {z_mae:.6f}")
        
        # calculate important points error
        important_points = [2, 4]
        true_important = tf.gather(true_reshaped, important_points, axis=1)
        pred_important = tf.gather(pred_reshaped, important_points, axis=1)
        
        important_mae = tf.reduce_mean(tf.abs(true_important - pred_important))
        important_mse = tf.reduce_mean(tf.square(true_important - pred_important))
        
        print("\nimportant points error:")
        print(f"MSE: {important_mse:.6f}")
        print(f"MAE: {important_mae:.6f}")
        
        # visualize result
        fig = visualize_comparison(image, true_landmarks, pred_landmarks,
                                 title=f'{Path(model_path).stem}\nMSE: {mse:.6f}, MAE: {mae:.6f}')
        fig.savefig(f'{Path(model_path).stem}_results.png')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """main function"""
    # test all available models
    models_to_test = [
        'hand_landmark_model_float16.tflite',
        'hand_landmark_model.h5',
        'hand_landmark_model.keras'
    ]
    
    test_results = []
    
    for model_path in models_to_test:
        if not Path(model_path).exists():
            print(f"\nskipping non-existent model: {model_path}")
            test_results.append(False)
            continue
            
        if model_path.endswith('.tflite'):
            result = test_tflite_model(model_path)
        else:
            result = load_and_test_model(model_path)
        test_results.append(result)
    
    # print summary
    print("\ntesting results:", "passed" if all(test_results) else "failed")

if __name__ == "__main__":
    main() 