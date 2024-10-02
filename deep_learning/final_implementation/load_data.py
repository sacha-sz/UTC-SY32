import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

from config import INPUT_SIZE



# ----------------------------------------- CHARGEMENT DES DONNEES ---------------------------------------------- #

data_dir = 'dataset'
train_img_dir = os.path.join(data_dir, 'train', 'images')
train_lab_dir = os.path.join(data_dir, 'train', 'labels')
val_img_dir = os.path.join(data_dir, 'val', 'images')
val_lab_dir = os.path.join(data_dir, 'val', 'labels')

# listes des chemins des images
train_img_paths = [os.path.join(train_img_dir, img) for img in os.listdir(train_img_dir)]
val_img_paths = [os.path.join(val_img_dir, img) for img in os.listdir(val_img_dir)]

# listes des noms des images
train_list = os.listdir(train_img_dir)
val_list = os.listdir(val_img_dir)



def get_img_from_filename(img_filename):
    """
    Fonction pour charger une image à partir de son nom de fichier
    :param img_filename: le nom de l'image
    """

    if img_filename in train_list:
        img_path = os.path.join(train_img_dir, img_filename)
    else:
        img_path = os.path.join(val_img_dir, img_filename)
    return plt.imread(img_path)

def get_boxes_from_filename(img_filename):
    """
    Fonction pour charger les boites englobantes à partir du nom de l'image
    :param img_filename: le nom de l'image
    """

    if img_filename in train_list:
        lab_path = os.path.join(train_lab_dir, img_filename[:-4] + '.csv')
    else:
        lab_path = os.path.join(val_lab_dir, img_filename[:-4] + '.csv')

    with open(lab_path, 'r') as f:
        lines = f.readlines()
        if "\n" in lines: lines.remove("\n")
        
    boxes = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line.split(",")[:4])
        label = line.split(",")[4].strip()
        if label == 'ff':
            continue
        boxes.append((label, [x1, y1, x2, y2]))

    return boxes

def format_image(img, boxes):
    """
    Fonction pour formater l'image et les boites englobantes
    :param img: l'image
    :param boxes: les boites englobantes
    """

    height, width, _ = img.shape 
    max_size = max(height, width)
    r = max_size / INPUT_SIZE
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    new_image[0:new_height, 0:new_width, :] = resized

    new_boxes = []
    for label, box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        w = x2 - x1
        h = y2 - y1
        # redimensionner la boîte
        x = int(x1 / r)
        y = int(y1 / r)
        w = int(w / r)
        h = int(h / r)
        new_box = [x, y, w, h]
        new_boxes.append((label, new_box))

    return new_image, new_boxes

def get_mask_from_boxes(boxes):
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE, 1), dtype=np.uint8)
    for label, box in boxes:
        x, y, w, h = box
        mask[y:y+h, x:x+w] = 1
    return mask

def data_load(files):
    """
    Fonction pour charger les données
    :param files: les fichiers
    """
    X = np.zeros((len(files), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    Y = np.zeros((len(files), INPUT_SIZE, INPUT_SIZE, 1), dtype=np.bool_)

    for i, file in enumerate(files):
        img = get_img_from_filename(file)
        boxes = get_boxes_from_filename(file)
        img, boxes = format_image(img, boxes)
        # img = img.astype(float) / 255.
        mask = get_mask_from_boxes(boxes)
        X[i]=img
        Y[i] = mask
    return X, Y

def get_data(files):
    lot_imgs = []
    lot_boxes = []

    for file in files:
        img = get_img_from_filename(file)
        boxes = get_boxes_from_filename(file)
        img, boxes = format_image(img, boxes)

        lot_imgs.append(img)
        lot_boxes.append(boxes)

    return lot_imgs, lot_boxes
