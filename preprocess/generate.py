import pandas as pd
import os
from PIL import Image
import numpy as np
from datetime import datetime

from detect_car import car_image

DATA_PATH = 'Myauto_data/Car_Images/{}/'
CSV_PATH = 'Myauto_data/MyAuto_ge_Cars_Data.csv'
CAR_IMAGE_DF = 'CAR_IMAGE_DF.csv'


def get_car_images(path):
    """
    Generate dataframe with columns: (ID, images)
    containing indexes of car exterior images for each car ID

    Args:
        path (str): Path to car features dataframe csv file
    Returns:
    (pd.DataFrame): Dataframe of ID's and images
    """
    data = pd.read_csv(path)
    car_image_df = pd.DataFrame({'ID': [], 'images': []})
    for i in range(len(data)):
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
            except:
                pass
        if len(images) != 0:
            images = images[:-1]
        car_image_df = car_image_df.append(
            pd.DataFrame({'ID': [int(ID)], 'images': [images]}))
    car_image_df.to_csv(CAR_IMAGE_DF, index=False)


def merge_car_features_and_car_images(data_path, image_data_path):
    """
    Merge car data with car images data

    Args:
        data_path (str): Data path
        image_data_path (str): Car images data path
    """
    car_data = pd.read_csv(image_data_path)
    full_data = pd.read_csv(data_path)
    car_data = car_data.drop_duplicates('ID')
    full_data = full_data.drop_duplicates('ID')
    merged = pd.merge(full_data, car_data, on='ID', how='right')
    return merged


def _get_number_of_images(id):
    """
    Helper function intended to be used in apply function.
    returns the number of images for car having the given ID

    Args:
        id (int): ID of the car
    Returns:
        (int): Number of images
    """
    try:
        images = os.listdir("/content/Car_Images/Car_Images/{}".format(id))
    except:
        return 0
    return len(images)


def _get_path(a):
    """
    Returns image path in a folder structure

    Args:
        a: Image data dataframe row
    Returns:
        (str): Generated path
    """
    return '/'+a.ID+'/'+a.img_index


def generate_image_oriented_dataset(df):
    """
    Generates new dataset for each image containing
    the same features as a car

    Args:
        df (pd.DataFrame): Car features dataframe
    Returns:
        (pd.DataFrame): Dataframe
    """
    df['img_num'] = df.ID.apply(_get_number_of_images)

    tmp_list = {'ID': [], 'Price ($)': [], 'Levy ($)': [], 'Manufacturer': [], 'Model': [], 'Prod. year': [],
               'Category': [], 'Leather interior': [], 'Fuel type': [], 'Engine volume': [], 'Mileage': [],
                'Cylinders': [], 'Gear box type': [], 'Drive wheels': [], 'Doors': [], 'Wheel': [], 'Color': [],
                'Interior color': [], 'Airbags': [], 'VIN': [], 'img_index': [], 'is_car': []}

   for i in df.index:
        curr_row = df.iloc[i]
        try:
            ls = [int(x) for x in curr_row.images.split('_')]
        except:
            try:
                ls = int(curr_row.images)
            except:
                continue
        for j in range(1, curr_row.img_num+1):
            for col in df.columns[1:-2]:
                tmp_list[col].append(curr_row[col])
            tmp_list['ID'].append(str(curr_row['ID']))
            tmp_list['img_index'].append(str(j))
            tmp_list['is_car'].append('1' if j in ls else '0')

    img_data = pd.DataFrame(tmp_list)
    img_data['path'] = img_data.apply(_get_path, axis=1)
    return img_data
