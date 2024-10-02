import tensorflow as tf
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from config import NB_CLASSES, PANNEAU_SIZE, CLASS_NAMES, INPUT_SIZE

from models.segmentation_model import unet_model
from load_data import data_load
from models.classification_model import panneau_model


# suppression du fichier detections.csv
if os.path.exists('detections.csv'):
    os.remove('detections.csv')


# ----------------------------------------- CHARGEMENT DU MODELE ---------------------------------------------- #

print("Chargement du modèle de segmentation et des poids ...")
model = unet_model()
model.load_weights('./deep_learning/final_implementation/checkpoints/segmodel.weights.h5')

print("Chargement du modèle de classification et des poids...")
model_panneau=panneau_model()
model_panneau.load_weights('./deep_learning/final_implementation/checkpoints/classmodel.weights.h5')


# ----------------------------------------- CHARGEMENT DES DONNEES ---------------------------------------------- #

data_dir = 'dataset'
set = os.path.join(data_dir, 'test')
files = [os.path.join(set, img) for img in os.listdir(set)]


# ----------------------------------------- VISUALISATION DES DONNEES ---------------------------------------------- #

# Récupérer les boxes des endroits des masques
def get_boxes(mask):
    mask = np.squeeze(mask)
    # plt.imshow(mask)
    # plt.colorbar()
    # plt.show()
    mask2 = mask.copy()
    mask2 = (mask2 > 0.3).astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    scores = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        boxes.append((x, y, x+w, y+h))
        # scores de confiance
        scores.append(np.mean(mask[y:y+h, x:x+w]))
    return boxes, scores

def get_label(img, box):
    pass


# ----------------------------------------- PREDICTION DE LA CLASSE DES PANNEAUX ---------------------------------------------- #

# Récupérer les images des panneaux
def get_panneaux(img, boxes, scores):
    panneaux = []
    s = []
    coord = []
    for box, score in zip(boxes, scores):
        x, y, x2, y2 = box
        panneau = img[y:y2, x:x2]
        if panneau.shape[0] < 10 or panneau.shape[1] < 10:
            continue
        # print("Panneau shape : ", panneau.shape)
        panneaux.append(panneau)
        s.append(score)
        coord.append(box)
    return panneaux, coord, s

def resize_img(img):
    resized_image = cv.resize(img, (PANNEAU_SIZE, PANNEAU_SIZE), interpolation=cv.INTER_LANCZOS4)
    if img.shape[0] > PANNEAU_SIZE or img.shape[1] > PANNEAU_SIZE:
        resized_image = cv.GaussianBlur(resized_image, (5,5), 0)
    return resized_image


# Pour chaque image du set 
for file in files : 
    
    # garder que le numéro de l'image dataset\test\0313.jpg
    num_img = file.split("\\")[-1][:-4]
    print("Nom du fichier : ", num_img)


    img = plt.imread(file)
    height, width, _ = img.shape 
    max_size = max(height, width)
    r = max_size / INPUT_SIZE
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    new_image[0:new_height, 0:new_width, :] = resized

    
    # Prédire la segmentation
    pred = model.predict(np.expand_dims(new_image, axis=0))
    
    # Récupérer les boxes
    boxes, scores = get_boxes(pred)

    # Récupérer les panneaux
    panneaux, coord, scores= get_panneaux(new_image, boxes, scores)
    panneaux = [resize_img(panneau) for panneau in panneaux]

    if len(panneaux) > 0:
        # prédire la classe des panneaux
        panneaux = np.array(panneaux, dtype=np.float32) /255
        predictions = model_panneau.predict(panneaux)
        
        # créer le fichiers detections.csv
        with open('detections.csv', 'a') as f:

            # Afficher les panneaux avec leur classe prédite
            for i, panneau in enumerate(panneaux):
                score_conf = np.max(predictions[i]*scores[i])
                if score_conf < 0.4:
                    continue
                # afficher un rectangle autour du panneau
                x, y, x2, y2 = coord[i]
                cv.rectangle(new_image, (x, y), (x2, y2), (0, 255, 0), 2)
                # afficher le nom de la classe et score sur l'image
                cv.putText(new_image, f"{CLASS_NAMES[np.argmax(predictions[i])]} : {score_conf * 100 :.1f}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

                print("Classe prédite : ", CLASS_NAMES[np.argmax(predictions[i])])
                print("Coordonnées : ", coord[i])
                print("Score de confiance : ", np.max(predictions[i]*scores[i]))

                # écrire dans le fichier detections.csv
                # mettre les coordonnées dans le vrai ordre de grandeur de l'image
                x, y, x2, y2 = int(x * r), int(y * r), int(x2 * r), int(y2 * r)
                f.write(f"{int(num_img)}, {x}, {y}, {x2}, {y2}, {score_conf}, {CLASS_NAMES[np.argmax(predictions[i])]}\n")
                
            # Afficher l'image avec matplotlib en rgb
            plt.imshow(new_image)        
            plt.show()






