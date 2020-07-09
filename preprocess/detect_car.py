import tensorflow.compat.v1 as tf
import cv2
from PIL import Image
import numpy as np
from datetime import datetime

# tf.disable_v2_behavior()

MOBILENET_SSD_PATH = 'models/mobilenet_ssd.tflite'

COCO_INP_SIZE = 300
CAR_ID = 2
CAR_THRESHOLD = 0.25


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

interpreter, input_details, output_details = load_tflite_model(
    MOBILENET_SSD_PATH)

def car_image(img):
    """
    Check if car is on the image

    Args:
        img (np.array): Image array
    
    Returns:
        (bool): True/False whether it's a car image or not
    """
    boxes, confidence, classes, nums = coco_detection(img)
    # Go through the prediction results
    for i in range(int(nums)):
        # Iterate over the predicted bounding boxes and filter the boxes with class "person"
        if classes[i] == CAR_ID and confidence[i] > 0.60:
            x1 = boxes[i][1]
            y1 = boxes[i][0]
            x2 = boxes[i][3]
            y2 = boxes[i][2]
            if (x2-x1)*(y2-y1) > CAR_THRESHOLD:
                return True
    return False


def coco_detection(img):
    """
    People detection with ssd mobilenet coco

    Args:
        img (np.array): Input image
   
    Returns:
        (list): list of human bounding boxes    
    """
    # Resize the image to size [1, 300, 300, 3]
    img_in = tf.expand_dims(img, 0)
    img_in = tf.image.resize(img_in, (COCO_INP_SIZE, COCO_INP_SIZE))
    img_in = tf.cast(img_in, dtype=tf.uint8)

    # Set img_in as tensor in the model's input_details
    interpreter.set_tensor(input_details[0]['index'], img_in)
    interpreter.invoke()

    # Get the output_details tensors (based on the given input above)
    bbox = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    nums = interpreter.get_tensor(output_details[3]['index'])

    boxes, confidence, classes, nums = bbox[0], scores[0], classes[0], nums[0]

    return boxes, confidence, classes, nums
