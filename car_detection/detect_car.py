import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np

from utils import load_tflite_model

# tf.disable_v2_behavior()

MOBILENET_SSD_PATH = 'models/mobilenet_ssd.tflite'

COCO_INP_SIZE = 300
CAR_ID = 2
CAR_THRESHOLD = 0.25

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

print(car_image(np.array(Image.open("/home/misho/Downloads/car.jpg"))))
