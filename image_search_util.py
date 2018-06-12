# make a Python list of the files in the directory
# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize

def find_files(directory):
    objects = os.listdir(directory)  # find all objects in a dir

    files = []
    for i in objects:  # check if very object in the folder ...
        if is_file(directory + i):  # ... is a file.
            files.append(i)  # if yes, append it.
    return files

def is_file(object):
    try:
        os.listdir(object)  # tries to get the objects inside of this object
        return False  # if it worked, it's a folder
    except Exception:  # if not, it's a file
        return True
    
def extract_feature_vector(model, input_img):
    input_img = cv2.resize(input_img, (224,224))
    features = model.predict([[input_img]])[0].reshape(49,1920)
    return features