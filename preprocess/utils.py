import tensorflow.compat.v1 as tf
import cv2

def load_tflite_model(tf_model_path):
    """
    Loads tflite model from a given path

    Args:
        tf_model_path (str): Path to the model
    
    Returns:
        (interpreter, input_details, output_details): 
        tf.lite.Interpreter, interpreter's input and output details 
    """
    # Load the tfline model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tf_model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details
