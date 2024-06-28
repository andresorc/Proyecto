#parámetros de entrada
# image_l: imagen izquierda
# image_r: imagen derecha
# input_masks : lista de máscaras de entrada

#parámetros de salida
# output_masks : lista de máscaras de salida, ifual a input_masks pero con el parámetro 'contorno' añadido

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Función que recorta una imagen según una región de interés
def recortar_imagen(imagen, bbox):
    x, y, w, h = bbox
    return imagen[y:y+h, x:x+w]


def convertir_a_json(objeto):
    if isinstance(objeto, np.integer):
        return int(objeto)
    elif isinstance(objeto, np.floating):
        return float(objeto)
    elif isinstance(objeto, np.ndarray):
        return objeto.tolist()
    elif isinstance(objeto, (list, dict, str, int, float, bool, type(None))):
        return objeto
    else:
        raise TypeError(f"Tipo {type(objeto)} no es serializable")




# Función que procesa las imágenes de entrada y añade el parámetro 'contorno' a las máscaras de salida
def procesar_imagenes(image_l, image_r, input_masks, mostrar=True):
    masks = input_masks
    for mask in masks:
        bbox_l = mask['l']['bbox']
        bbox_r = mask['r']['bbox']
        dx_l = bbox_l[0]
        dy_l = bbox_l[1]
        dx_r = bbox_r[0]
        dy_r = bbox_r[1]
        mask_l = recortar_imagen(image_l, bbox_l)
        mask_r = recortar_imagen(image_r, bbox_r)
        #Mostrar imagenes
        if mostrar:
            plt.imshow(mask_l, cmap='gray')
            plt.title('Recorte izquierdo')
            plt.axis('off')
            plt.show()

            plt.imshow(mask_r, cmap='gray')
            plt.title('Recorte derecho')
            plt.axis('off')
            plt.show()
        # Encontrar bordes de las mascaras mediante umbralizaion dinamica en funcion del brilllo medio de la imagen
        #Aplicar umbralizacion
        mean_l = np.mean(mask_l)
        mean_r = np.mean(mask_r)
        
        #Aplicar umbralizacion por Otsu
        _, mask_l = cv2.threshold(mask_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_r = cv2.threshold(mask_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #hallar coordenadas de los bordes de canny
        #coords_l = np.column_stack(np.where(mask_l > 0))
        #coords_r = np.column_stack(np.where(mask_r > 0))

        

        #Mostar las imágenes con los bordes
        if mostrar:
            plt.imshow(mask_l, cmap='gray')
            plt.title('Recorte izquierdo')
            plt.axis('off')
            plt.show()

            plt.imshow(mask_r, cmap='gray')
            plt.title('Recorte derecho')
            plt.axis('off')
            plt.show()

        # Encontrar contornos en las máscaras
        contours_l,_ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_r,_ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #Encontrar el contorno con el área más grande
        best_cnt_l = max(contours_l, key=cv2.contourArea)
        best_cnt_r = max(contours_r, key=cv2.contourArea)

        mask_l = cv2.cvtColor(mask_l, cv2.COLOR_GRAY2BGR)
        mask_r = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_l, best_cnt_l, -1, (0, 255, 0), 2)
        cv2.drawContours(mask_r, best_cnt_r, -1, (0, 255, 0), 2)

        if mostrar:
            cv2.imshow('Recorte izquierdo con mejor contorno', mask_l)
            cv2.imshow('Recorte derecho con mejor contorno', mask_r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #Calcular coordenadas originales
        cnt_l = best_cnt_l + np.array([dx_l, dy_l])
        cnt_r = best_cnt_r + np.array([dx_r, dy_r])
        mask['l']['cnt'] = cnt_l.tolist()
        mask['r']['cnt'] = cnt_r.tolist()
    return masks

#Lectura del sufijo de la imagen
with open('Images/Pruebas/sufijo.txt', 'r') as f:
    sufijo = f.read().strip()

#Entrada
image_l = cv2.imread(f'Images/Pruebas/Corrected/cam_sup_l_corrected_{sufijo}.jpg')
image_r = cv2.imread(f'Images/Pruebas/Corrected/cam_sup_r_corrected_{sufijo}.jpg')
#ecualizar imagenes
image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
ruta_masks = 'masks/masks_filt_morf.json'

#Salida
ruta_output_masks = 'masks/masks_contours.json'

input_masks = []
with open(ruta_masks, 'r') as archivo:
    input_masks = json.load(archivo)

output_masks = procesar_imagenes(image_l, image_r, input_masks, mostrar = False)

with open(ruta_output_masks, 'w') as archivo_json:
    json.dump(output_masks, archivo_json, default=convertir_a_json, indent=4)
