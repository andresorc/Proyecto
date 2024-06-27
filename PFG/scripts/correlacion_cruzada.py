import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Función para recortar la porción de imagen definida por la bounding box
def recortar_porcion_imagen(imagen, bbox):
    x, y, w, h = bbox
    return imagen[y:y + h, x:x + w]

# Función para calcular la correlación cruzada entre dos imágenes
def calcular_correlacion_cruzada(imagen1, imagen2):
    # Convertir imágenes a escala de grises (si no lo están ya)
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)

    # Normalizar las imágenes para asegurar que los valores estén en el rango [0, 1]
    imagen1 = cv2.normalize(imagen1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    imagen2 = cv2.normalize(imagen2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Calcular la correlación cruzada utilizando scipy.signal.correlate2d
    correlacion = correlate2d(imagen1, imagen2, mode='same')

    # Calcular un valor único promedio representativo de la similitud total
    promedio = np.mean(correlacion)

    return correlacion, promedio


# Cargar la imagen
imagen_l = cv2.imread('Images/FrameI_movil_corrected.jpg')
imagen_r = cv2.imread('Images/FrameD_movil_corrected.jpg')

ruta_bounding_boxes_frame1 = 'masks/masks_l.json'
ruta_bounding_boxes_frame2 = 'masks/masks_r.json'

# Cargar las bounding boxes desde los archivos JSON
with open(ruta_bounding_boxes_frame1, 'r') as archivo:
    bounding_boxes_frame1 = json.load(archivo)

with open(ruta_bounding_boxes_frame2, 'r') as archivo:
    bounding_boxes_frame2 = json.load(archivo)

# Seleccionar las bounding boxes de interés
bbox1 = bounding_boxes_frame1[6]
bbox2 = bounding_boxes_frame2[12]

# Dibujar las bounding boxes en las imágenes originales
cv2.rectangle(imagen_l, (bbox1[0], bbox1[1]), (bbox1[2]+bbox1[0], bbox1[3]+bbox1[1]), (0, 255, 0), 2)
cv2.rectangle(imagen_r, (bbox2[0], bbox2[1]), (bbox1[2]+bbox1[0], bbox1[3]+bbox1[1]), (0, 255, 0), 2)

# Recortar las porciones de imagen correspondientes a las bounding boxes
porcion1 = recortar_porcion_imagen(imagen_l, bbox1)
porcion2 = recortar_porcion_imagen(imagen_r, bbox2)

# Calcular la correlación cruzada entre las dos porciones de imagen
resultado_correlacion,promedio = calcular_correlacion_cruzada(porcion1, porcion2)

# Imprimir el resultado de la correlación cruzada
print("Promedio de la correlación cruzada:")
print(promedio)

# Mostrar las imágenes con bounding boxes y las porciones recortadas
cv2.imshow('Imagen L con Bounding Box', imagen_l)
cv2.imshow('Imagen R con Bounding Box', imagen_r)
cv2.imshow('Porción 1', porcion1)
cv2.imshow('Porción 2', porcion2)

# Visualizar la matriz de correlación cruzada
plt.imshow(resultado_correlacion, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Correlación Cruzada')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()