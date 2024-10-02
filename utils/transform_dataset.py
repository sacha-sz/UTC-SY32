import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def clean(lines):
    """
    Nettoie les lignes des labels en ne gardant que les valeurs importantes : x1, x2, x3, x4, label.
    Supprime les espaces et les retours à la ligne.

    param:
    - lines : liste des lignes du fichier

    return:
    - liste des lignes nettoyées
    """
    
    cleaned_lines = []
    
    for line in lines:
        line = line.strip().replace(" ", "")
        if line and line[-2:].upper() != "FF":
            cleaned_lines.append(line)
    
    return cleaned_lines

def intersection_over_union(coord, coord_label):
    """
    Calcule le rapport de l'intersection sur l'union (IoU) entre deux rectangles définis par leurs coordonnées.

    coord : (x_haut_gauche, y_haut_gauche, x_bas_droite, y_bas_droite)
    coord_label : (x_haut_gauche, y_haut_gauche, x_bas_droite, y_bas_droite)

    return:
    - IoU (float)
    """
    x1, y1, x2, y2 = coord
    x1_l, y1_l, x2_l, y2_l = coord_label

    # Calcul de l'intersection
    x_inter_1 = max(x1, x1_l)
    y_inter_1 = max(y1, y1_l)
    x_inter_2 = min(x2, x2_l)
    y_inter_2 = min(y2, y2_l)

    inter_width = max(0, x_inter_2 - x_inter_1 + 1)
    inter_height = max(0, y_inter_2 - y_inter_1 + 1)
    interArea = inter_width * inter_height

    # Calcul des aires des rectangles
    boxArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxArea_label = (x2_l - x1_l + 1) * (y2_l - y1_l + 1)

    # Calcul de l'IoU
    iou = interArea / float(boxArea + boxArea_label - interArea)

    return iou



if __name__ == "__main__":
    train_img = os.path.join("dataset", "train", "images")
    train_lab = os.path.join("dataset", "train", "labels")
    train_list = os.listdir(train_img)

    val_img = os.path.join("dataset", "val", "images")
    val_lab = os.path.join("dataset", "val", "labels")
    val_list = os.listdir(val_img)


    path_new_dataset = os.path.join("full_dataset")
    os.makedirs(path_new_dataset, exist_ok=True)

    subfolders = ["frouge", "fvert", "forange", "interdiction", "danger", "stop", "ceder", "obligation", "none"]
    association = {
        "frouge": 0,
        "fvert": 0,
        "forange": 0,
        "interdiction": 0,
        "danger": 0,
        "stop": 0,
        "ceder": 0,
        "obligation": 0,
        "none": 0
    }

    for subfolder in subfolders:
        os.makedirs(os.path.join(path_new_dataset, subfolder), exist_ok=True)
        association[subfolder] = os.path.join(path_new_dataset, subfolder)


    seuil = 0.4
    taille = 64
    nb_neg_par_img = 2

    neg_compteur = 0

    for img in train_list :
        img_path = os.path.join(train_img, img)
        lab_path = os.path.join(train_lab, img[:-4] + ".csv")
                
        img_data = cv2.imread(img_path)
        img_h, img_w, _ = img_data.shape

        with open(lab_path, "r") as f:
            lines = clean(f.readlines())

            uuid = 0
            for x1, y1, x2, y2, label in [line.split(",") for line in lines]:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                before_cropped = img_data[y1:y2, x1:x2]

                if before_cropped.shape < (taille, taille, 3):
                    before_cropped = cv2.resize(before_cropped, (taille, taille), interpolation=cv2.INTER_LANCZOS4)
                elif before_cropped.shape > (taille, taille, 3):
                    before_cropped = cv2.resize(before_cropped, (taille, taille), interpolation=cv2.INTER_AREA)


                path = os.path.join(association[label], f"{img[:-4]}_{uuid}.jpg")
                if not os.path.exists(path):
                    cv2.imwrite(path, before_cropped)
                uuid += 1

                # Data Augmentation
                if label != "interdiction":
                    # Ajout de FLIPLR
                    flip_lr = cv2.flip(before_cropped, 1)
                    path = os.path.join(association[label], f"{img[:-4]}_{uuid}.jpg")
                    if not os.path.exists(path):
                        cv2.imwrite(path, flip_lr)
                    uuid += 1

                    # Ajout d'une luminosité aléatoire
                    alpha = 1.5 + (random.random() - 0.5)
                    beta = 0
                    brightness = cv2.convertScaleAbs(before_cropped, alpha=alpha, beta=beta)
                    path = os.path.join(association[label], f"{img[:-4]}_{uuid}.jpg")
                    if not os.path.exists(path):
                        cv2.imwrite(path, brightness)
                    uuid += 1
                    
                aire_libre = img_h * img_w - sum([(int(x2) - int(x1)) * (int(y2) - int(y1)) for x1, y1, x2, y2, _ in [line.split(",") for line in lines]])
                if aire_libre / (img_h * img_w) < 0.6:
                    # Si on ne peut pas générer d'images négatives
                    continue    
                
                for _ in range(nb_neg_par_img):
                    rect_or_square = np.random.randint(0, 2)
                    if rect_or_square == 0:
                        # On tire un rectangle donc w < h
                        w = taille * np.random.uniform(0.5, 1.5)
                        h = taille * np.random.uniform(1, 1.5)
                    else:
                        # On tire un carré
                        w = taille * np.random.uniform(0.5, 1.5)
                        h = w

                    x1_new = np.random.randint(0, img_w - w)
                    y1_new = np.random.randint(0, img_h - h)
                    x2_new = x1_new + w
                    y2_new = y1_new + h

                    if len(lines) != 0:
                        # S'il existe des panneaux on vérifie les intersections
                        while any([intersection_over_union((x1_new, y1_new, x2_new, y2_new), (int(x1), int(y1), int(x2), int(y2))) > seuil for x1, y1, x2, y2, _ in [line.split(",") for line in lines]]):
                            x1_new = np.random.randint(0, img_w - w)
                            y1_new = np.random.randint(0, img_h - h)
                            x2_new = x1_new + w
                            y2_new = y1_new + h

                    cropped_image = img_data[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
                    if cropped_image.shape < (taille, taille, 3):
                        cropped_image = cv2.resize(cropped_image, (taille, taille), interpolation=cv2.INTER_LANCZOS4)

                    if cropped_image.shape > (taille, taille, 3):
                        cropped_image = cv2.resize(cropped_image, (taille, taille), interpolation=cv2.INTER_AREA)

                    # On ajoute l'image dans le dossier correspondant
                    if not os.path.exists(os.path.join(association["none"], f"{img[:-4]}_{neg_compteur}.jpg")):
                        cv2.imwrite(os.path.join(association["none"], f"{img[:-4]}_{neg_compteur}.jpg"), cropped_image)
                    neg_compteur += 1
                
    nb_images = {}
    for folder in association.keys():
        nb_images[folder] = len(os.listdir(association[folder]))
        
    plt.figure()
    plt.bar(nb_images.keys(), nb_images.values())
    plt.xticks(rotation=90)
    plt.show()