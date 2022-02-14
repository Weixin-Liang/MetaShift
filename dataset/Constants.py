

import os
import json
import copy
import random
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Folders
IMAGE_DATA_FOLDER = '/data/GQA/allImages/images/' # replace with your path

ONLY_SELECTED_CLASSES = True # Use False to generate the MetaDataset for all ~400 classes [Warning: Very Large]. 
SELECTED_CLASSES = [
    'cat', 'dog',
    'bus', 'truck',
    'elephant', 'horse',
    ] 

with open('./meta_data/class_hierarchy.json') as f:
    class_hierarchy = json.load(f)

ATTRIBUTE_CONTEXT_ONTOLOGY = {
 'darkness': ['dark', 'bright'],
 'dryness': ['wet', 'dry'],
 'colorful': ['colorful', 'shiny'],
 'leaf': ['leafy', 'bare'],
 'emotion': ['happy', 'calm'],
 'sports': ['baseball', 'tennis'],
 'flatness': ['flat', 'curved'],
 'lightness': ['light', 'heavy'],
 'gender': ['male', 'female'],
 'width': ['wide', 'narrow'],
 'depth': ['deep', 'shallow'],
 'hardness': ['hard', 'soft'],
 'cleanliness': ['clean', 'dirty'],
 'switch': ['on', 'off'],
 'thickness': ['thin', 'thick'],
 'openness': ['open', 'closed'],
 'height': ['tall', 'short'],
 'length': ['long', 'short'],
 'fullness': ['full', 'empty'],
 'age': ['young', 'old'],
 'size': ['large', 'small'],
 'pattern': ['checkered', 'striped', 'dress', 'dotted'],
 'shape': ['round', 'rectangular', 'triangular', 'square'],
 'activity': ['waiting', 'staring', 'drinking', 'playing', 'eating', 'cooking', 'resting', 
              'sleeping', 'posing', 'talking', 'looking down', 'looking up', 'driving', 
              'reading', 'brushing teeth', 'flying', 'surfing', 'skiing', 'hanging'],
 'pose': ['walking', 'standing', 'lying', 'sitting', 'running', 'jumping', 'crouching', 
            'bending', 'smiling', 'grazing'],
 'material': ['wood', 'plastic', 'metal', 'glass', 'leather', 'leather', 'porcelain', 
            'concrete', 'paper', 'stone', 'brick'],
 'color': ['white', 'red', 'black', 'green', 'silver', 'gold', 'khaki', 'gray', 
            'dark', 'pink', 'dark blue', 'dark brown',
            'blue', 'yellow', 'tan', 'brown', 'orange', 'purple', 'beige', 'blond', 
            'brunette', 'maroon', 'light blue', 'light brown']
}


GENERAL_CONTEXT_ONTOLOGY = {
    'location': ['indoors', 'outdoors'],
    'weather': ['clear', 'overcast', 'cloudless', 'cloudy', 'sunny', 'foggy', 'rainy'],
    'room': ['bedroom', 'kitchen', 'bathroom', 'living room'],
    'place': ['road', 'sidewalk', 'field', 'beach', 'park', 'grass', 'farm', 'ocean', 'pavement',
            'lake', 'street', 'train station', 'hotel room', 'church', 'restaurant', 'forest', 'path',
            'display', 'store', 'river', 'sea', 'yard', 'airport', 'parking lot']
}


