import pandas as pd
import os
from PIL import Image
import numpy as np
from datetime import datetime

from detect_car import car_image

DATA_PATH = 'Myauto_data/Car_Images/{}/'
CSV_PATH = 'Myauto_data/MyAuto_ge_Cars_Data.csv'
CAR_IMAGE_DF = 'CAR_IMAGE_DF.csv'

def get_car_images():
    """
    Generate csv file with columns: (ID, images)
    containing indexes of car exterior images for each car ID
    """
    data = pd.read_csv(CSV_PATH)
    car_image_df = pd.DataFrame({'ID':[], 'images':[]})
    for i in range(0, 100):
        ID = data.iloc[i].ID
        images = ''
        try:
            image_dir = os.listdir(DATA_PATH.format(ID))
        except:
            continue
        for img in image_dir:
            try:
                if car_image(np.array(Image.open(DATA_PATH.format(ID)+img))):
                    images += img.split('.')[0] + '_'
            except: pass
        if len(images) != 0:
            images = images[:-1]
        car_image_df = car_image_df.append(pd.DataFrame({'ID':[int(ID)], 'images':[images]}))
    car_image_df.to_csv(CAR_IMAGE_DF, index=False)
