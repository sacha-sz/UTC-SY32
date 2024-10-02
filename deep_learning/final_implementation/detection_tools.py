import numpy as np
import cv2 as cv
import multiprocessing
import random


def bruit(image_orig):
    h, w, c = image_orig.shape
    n = np.random.randn(h, w, c) * random.randint(2, 15)  # Reduced noise intensity
    return np.clip(image_orig + n, 0, 255).astype(np.uint8)

def change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


def modif_img(image, boxes):
    h, w, c = image.shape
    img = image.copy()

    # Ajout de flou (1 fois sur 3)
    if np.random.randint(4):
        k_max = 2  
        kernel_blur = np.random.randint(1, k_max * 2 + 1)  
        if kernel_blur % 2 == 0:
            kernel_blur += 1
        img = cv.GaussianBlur(img, (kernel_blur, kernel_blur), 0)

    # # Ajout de rotation (1 fois sur 3)
    # if np.random.randint(3):
    #     print("rotation")
    #     angle = np.random.randint(-10, 10)
    #     M = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    #     img = cv.warpAffine(img, M, (w, h))
        
    #     # Modification des boîtes pour la rotation
    #     new_boxes = []
    #     for label, box in boxes:
    #         x1, y1, x2, y2 = box
    #         corners = np.array([
    #             [x1, y1],
    #             [x2, y1],
    #             [x2, y2],
    #             [x1, y2]
    #         ])
    #         ones = np.ones(shape=(len(corners), 1))
    #         corners_ones = np.hstack([corners, ones])
    #         transformed_corners = M.dot(corners_ones.T).T
    #         x_coords = transformed_corners[:, 0]
    #         y_coords = transformed_corners[:, 1]
    #         x1_new, y1_new, x2_new, y2_new = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    #         new_boxes.append((label, [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]))
    #     boxes = new_boxes

    # # Ajout de perspective (1 fois sur 4)
    # if np.random.randint(5):
    #     print("perspective")
    #     pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    #     pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    #     for i in range(4):
    #         pts2[i] = [pts2[i][0] + np.random.randint(-10, 10), pts2[i][1] + np.random.randint(-10, 10)]
    #     M = cv.getPerspectiveTransform(pts1, pts2)
    #     img = cv.warpPerspective(img, M, (w, h))

    #     # Modification des boîtes pour la perspective
    #     new_boxes = []
    #     for label, box in boxes:
    #         x1, y1, x2, y2 = box
    #         corners = np.array([
    #             [x1, y1],
    #             [x2, y1],
    #             [x2, y2],
    #             [x1, y2]
    #         ], dtype='float32')
    #         transformed_corners = cv.perspectiveTransform(np.array([corners]), M)[0]
    #         x_coords = transformed_corners[:, 0]
    #         y_coords = transformed_corners[:, 1]
    #         x1_new, y1_new, x2_new, y2_new = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    #         new_boxes.append((label, [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]))
    #     boxes = new_boxes

    # # Flip right/left
    # if np.random.randint(2):
    #     print("flip")
    #     img = cv.flip(img, 1)
    #     boxes = [(label, [w - x2, y1, w - x1, y2]) for label, (x1, y1, x2, y2) in boxes]

    # Changer luminosité 
    img=change_gamma(img, random.uniform(0.6, 1.0), -np.random.randint(50))

    img=bruit(img)

    return img, boxes

def create_lot_img(image, boxes, nbr):
    lot_img = []
    lot_boxes = []
    for _ in range(nbr):
        img, box = modif_img(image, boxes)
        lot_img.append(img)
        lot_boxes.append(box)
    return lot_img, lot_boxes

def augment_images_chunk(chunk):
    list_images, list_boxes, nbr_augmentation = chunk
    augmented_images = []
    augmented_boxes = []

    for k in range(len(list_images)):
        image = list_images[k]
        boxes = list_boxes[k]
        lot_img, lot_boxes = create_lot_img(image, boxes, nbr_augmentation)
        augmented_images.extend(lot_img)
        augmented_boxes.extend(lot_boxes)

    return augmented_images, augmented_boxes

def augment_dataset(train_images, train_boxes, nbr_augmentation, nbr_thread=None):
    if nbr_thread is None:
        nbr_thread = multiprocessing.cpu_count()

    chunk_size = (len(train_images) + nbr_thread - 1) // nbr_thread  # Adjusted to ensure all images are included
    chunks = [(train_images[i:i + chunk_size], train_boxes[i:i + chunk_size], nbr_augmentation)
              for i in range(0, len(train_images), chunk_size)]

    with multiprocessing.Pool(nbr_thread) as pool:
        results = pool.map(augment_images_chunk, chunks)

    augmented_images = []
    augmented_labels = []

    for images, labels in results:
        augmented_images.extend(images)
        augmented_labels.extend(labels)

    return train_images + augmented_images, train_boxes + augmented_labels
