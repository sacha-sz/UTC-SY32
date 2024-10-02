import os 
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from config import CLASSES, NB_CLASSES, PANNEAU_SIZE, BATCH_SIZE
from load_data import get_data
from detection_tools import augment_dataset_img_and_boxes
import models.classification_model as classification_model


print("+---------------------------------------------+")
print("|                                             |")
print("|           Entrainement du modèle            |")
print("|               CLASSIFICATION                |")
print("|                                             |")
print("+---------------------------------------------+")


NBR_AUGMENT = 0

def resize_lot_img(images):
    array = images.copy()
    print("Redimensionnement des images à une taille de ", PANNEAU_SIZE, "x", PANNEAU_SIZE, "...")
    for i, img in enumerate(images):
        resized_image = cv2.resize(img, (PANNEAU_SIZE, PANNEAU_SIZE), interpolation=cv2.INTER_LANCZOS4)
        if img.shape[0] > PANNEAU_SIZE or img.shape[1] > PANNEAU_SIZE:
            resized_image = cv2.GaussianBlur(resized_image, (5,5), 0)
        array[i] = resized_image

    return array

# ----------------------------------------- CREATION DES DONNEES D'ENTRAINEMENT ---------------------------------------------- #

print("Création des données d'entraînement ", end="")

if NBR_AUGMENT : 
    print("avec augmentation...")
else:
    print("sans augmentation...")

data_dir = '../../dataset'
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

train_images, train_boxes = get_data(train_list)
val_images, val_boxes = get_data(val_list)


# DATA AUGMENTATION
if NBR_AUGMENT:
    train_images, train_boxes = augment_dataset_img_and_boxes(train_images, train_boxes, NBR_AUGMENT)

X_train = []
Y_train = []
for img, boxes in zip(train_images, train_boxes):
    if len(boxes) > 0 : 
        for label, box in boxes:
            x, y, w, h = box
            if w < 10 or h < 10:
                continue
            cropped_image = img[y:y+h, x:x+w]
            X_train.append(cropped_image)
            Y_train.append(CLASSES[label])

X_train = resize_lot_img(X_train)
X_train = np.array(X_train, dtype=np.float32) /255
Y_train = np.array(Y_train, dtype = np.float32).reshape(-1, 1)


print("Création des données de validation ")

X_val = []
Y_val = []
for img, boxes in zip(val_images, val_boxes):
    if len(boxes) > 0 : 
        for label, box in boxes:
            x, y, w, h = box
            if w < 10 or h < 10:
                continue
            cropped_image = img[y:y+h, x:x+w]
            X_val.append(cropped_image)
            Y_val.append(CLASSES[label])


X_val = resize_lot_img(X_val)
X_val = np.array(X_val, dtype=np.float32) /255
Y_val = np.array(Y_val, dtype = np.float32).reshape(-1, 1)



    

# ----------------------------------------- VISUALISATION DES DONNEES ---------------------------------------------- #

index = random.randint(0, len(X_train))
plt.imshow(X_train[index])
plt.title(f"Donnée entrainement : {Y_train[index]}")
plt.show()

index = random.randint(0, len(X_val))
plt.imshow(X_val[index])
plt.title(f"Donnée validation : {Y_val[index]}")
plt.show()

# distribution des classes
plt.hist(Y_train, bins=NB_CLASSES)
plt.title("Distribution des classes")
plt.show()


# ----------------------------------------- ENTRAINEMENT DU MODELE ---------------------------------------------- #

print("Entrainement du modèle...")
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)

optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
train_loss=tf.keras.metrics.Mean()
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
test_loss=tf.keras.metrics.Mean()
test_accuracy=tf.keras.metrics.SparseCategoricalAccuracy()
model_panneau=classification_model.panneau_model()
checkpoint=tf.train.Checkpoint(model_panneau=model_panneau)

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions=model_panneau(images)
    loss=loss_object(labels, predictions)
  gradients=tape.gradient(loss, model_panneau.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model_panneau.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

def train(train_ds, nbr_entrainement):
  for entrainement in range(nbr_entrainement):
    start=time.time()
    for images, labels in train_ds:
      train_step(images, labels)
    message='Entrainement {:04d}: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(entrainement+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         time.time()-start))
    train_loss.reset_state()
    train_accuracy.reset_state()
    test(val_ds)
    
def test(test_ds):
  start=time.time()
  for test_images, test_labels in test_ds:
    predictions=model_panneau(test_images)
    t_loss=loss_object(test_labels, predictions)
    test_loss(t_loss)
    test_accuracy(test_labels, predictions)
  message='   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
  print(message.format(test_loss.result(),
                       test_accuracy.result()*100,
                       time.time()-start))
  test_loss.reset_state()
  test_accuracy.reset_state()

train(train_ds, 20)
model_panneau.save_weights("./deep_learning/final_implementation/checkpoints/classmodel.weights.h5")