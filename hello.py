import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger l'image
img = cv2.imread('/home/massilia/Workspace/LasqoApp/data/img/image copy 2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer une binarisation Otsu
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Suppression du bruit avec une ouverture morphologique
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Détermination de l'arrière-plan certain
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Détermination de l'avant-plan certain
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Détermination de la région inconnue
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marquage des composants
ret, markers = cv2.connectedComponents(sure_fg)

# Ajout de 1 à toutes les étiquettes pour que l'arrière-plan soit 1 au lieu de 0
markers = markers + 1

# Marquer la région inconnue avec zéro
markers[unknown == 255] = 0

# Application de l'algorithme de Watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# Affichage du résultat
cv2.imwrite("segmented_image.png", img)  # Sauvegarde l’image segmentée
print("✅ Image segmentée sauvegardée sous 'segmented_image.png'")
