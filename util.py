import tensorflow as tf
import numpy as np
import os
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.utils import shuffle

from variables import*

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img
    
def load_test_data(data_path, save_path):
    data_name = os.path.split(save_path)[-1].split('_')[0]
    if not os.path.exists(save_path):
        print("{} Images Saving".format(data_name))
        images = []
        classes = []
        url_strings = []
        dog_folders = os.listdir(data_path)
        for label in list(dog_folders):
            label_dir = os.path.join(data_path, label)
            label_images = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = preprocessing_function(img)
                img = cv.resize(img, target_size, cv.INTER_AREA).astype(np.float32)

                images.append(img)
                classes.append(int(label))
                url_strings.append(img_path)

        images = np.array(images).astype('float32')
        classes = np.array(classes).astype('float32')
        url_strings = np.array(url_strings)
        np.savez(save_path, name1=images, name2=classes, name3=url_strings)
    else:
        data = np.load(save_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']
        url_strings = data['name3']
        print("{} Images Loaded".format(data_name))

    classes, images, url_strings = shuffle(classes, images, url_strings)
    return classes, images, url_strings

def nearest_neighbour_prediction(result, test_classes):
    labels = [int(test_classes[neighbour_img_id]) for neighbour_img_id in result[1:]]
    label = np.bincount(labels).argmax()
    labels = np.array(labels)
    correct_idx = np.where(labels == label)[0]
    return result[correct_idx[:thres_neighbours+1]]