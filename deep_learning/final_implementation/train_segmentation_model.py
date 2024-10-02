import tensorflow as tf
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from models.segmentation_model import unet_model
from load_data import data_load
from tqdm import tqdm


NBR_AUGMENT = 0 # NE MARCHE PAS

print("+---------------------------------------------+")
print("|                                             |")
print("|           Entrainement du modèle            |")
print("|                SEGMENTATION                 |")
print("|                                             |")
print("+---------------------------------------------+")


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




from load_data import get_data
X_train, Y_train = get_data(train_list)
X_val, Y_val = get_data(val_list)


# ----------------------------------------- DATA AUGMENTATION ---------------------------------------------- #

from detection_tools import create_lot_img
from load_data import format_image, get_mask_from_boxes


if NBR_AUGMENT:
    print("Nombre d'images avant augmentation: ", len(X_train))
    X = []
    Y = []
    for i in tqdm(range(len(X_train))):
        X.append(X_train[i])
        Y.append(Y_train[i])
        lot_img, lot_lab = create_lot_img(X_train[i], Y_train[i], NBR_AUGMENT)
        X.extend(lot_img)
        Y.extend(lot_lab)

    print("Nombre d'images après augmentation: ", len(X))

    X_train = []
    Y_train = []
    for i in tqdm(range(len(X))):
        img, boxes = format_image(X[i], Y[i])
        mask = get_mask_from_boxes(boxes)
        X_train.append(img)
        Y_train.append(mask)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

else : 
    X_train, Y_train = data_load(train_list)
    X_val, Y_val = data_load(val_list)

        



# ----------------------------------------- VISUALISATION DES DONNEES ---------------------------------------------- #


# afficher une image aléatoire avec son masque sur la meme figure
ix = random.randint(0, len(X_train-1))
plt.imshow(X_train[ix])
plt.imshow(np.squeeze(Y_train[ix]), alpha=0.5)
plt.title('Image et masque')
plt.show()
# afficher une image aléatoire avec son masque sur la meme figure
ix = random.randint(0, len(X_train-1))
plt.imshow(X_train[ix])
plt.imshow(np.squeeze(Y_train[ix]), alpha=0.5)
plt.title('Image et masque')
plt.show()
# afficher une image aléatoire avec son masque sur la meme figure
ix = random.randint(0, len(X_train-1))
plt.imshow(X_train[ix])
plt.imshow(np.squeeze(Y_train[ix]), alpha=0.5)
plt.title('Image et masque')
plt.show()
# afficher une image aléatoire avec son masque sur la meme figure
ix = random.randint(0, len(X_train-1))
plt.imshow(X_train[ix])
plt.imshow(np.squeeze(Y_train[ix]), alpha=0.5)
plt.title('Image et masque')
plt.show()



# ----------------------------------------- ENTRAINEMENT DU MODELE ---------------------------------------------- #
checkpoint_model = tf.keras.callbacks.ModelCheckpoint('segmodel.keras', verbose=1, save_best_only=True)
checkpoint_weight = tf.keras.callbacks.ModelCheckpoint('./deep_learning/final_implementation/checkpoints/segmodel_aug.weights.h5',
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True, 
                                                save_weights_only=True,  
                                                mode='min')

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=[checkpoint_weight])


