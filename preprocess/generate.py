import pandas as pd
import os
from PIL import Image
import numpy as np
from datetime import datetime
import os

from preprocess.detect_car import car_image

DATA_PATH = '/home/misho/Uni/Vision/YOLO/keras-YOLOv3-model-set/Myauto_data/Car_Images/'
CSV_PATH = 'Myauto_data/MyAuto_ge_Cars_Data.csv'
CAR_IMAGE_DF = 'CAR_IMAGE_DF_FINAL.csv'


def get_car_images(data):
    """
    Generate dataframe with columns: (ID, images)
    containing indexes of car exterior images for each car ID

    Args:
        data (pd.DataFrame): Car features dataframe
    Returns:
    (pd.DataFrame): Dataframe of ID's and images
    """
    car_image_df = pd.DataFrame({'ID': [], 'images': []})
    for i in range(len(data)):
        ID = data.iloc[i].ID
        images = ''
        try:
            image_dir = os.listdir(DATA_PATH + '{}/'.format(ID))
        except:
            continue
        for img in image_dir:
            try:
                if car_image(np.array(Image.open(DATA_PATH + '{}/'.format(ID)+img))):
                    images += img.split('.')[0] + '_'
            except:
                pass
        if len(images) != 0:
            images = images[:-1]
        car_image_df = car_image_df.append(
            pd.DataFrame({'ID': [int(ID)], 'images': [images]}))

    return car_image_df


def merge_car_features_and_car_images(full_data, car_data):
    """
    Merge car data with car images data

    Args:
        full_data (pd.Dataframe): car features dataframe
        car_data (pd.Dataframe): Car images dataframe
    """
    car_data = car_data.drop_duplicates('ID')
    full_data = full_data.drop_duplicates('ID')
    merged = pd.merge(full_data, car_data, on='ID', how='left')
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
        images = os.listdir(DATA_PATH + '{}/'.format(id))
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

    tmp_list = {'ID': [], 'Category': [], 'Doors': [], 'Color': [], 'img_index': [], 'is_car': []}

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
    return img_data
    img_data['path'] = img_data.apply(_get_path, axis=1)
    return img_data


def copy_image(a, dir_name):
    try:
        img = Image.open(DATA_PATH + '{}/{}.jpg'.format(a.ID, a.img_index))
        img.save('{}/{}_{}.jpg'.format(dir_name, a.ID,a.img_index))
        return True
    except: 
        return False


def generate_training_folder(df, folder_name):
    try:
        os.mkdir(folder_name)
    except:
        pass
    img_success = []
    for i in range(len(df)):
        row = df.iloc[i]
        img_success.append(copy_image(row, folder_name))
    df['success'] = img_success
    return df

    
