import cv2
import numpy as np
import os
import shutil
import json
import matplotlib.pyplot as plt

def recortar_imagen(imagen, bbox):
    x, y, w, h = bbox
    return imagen[y:y+h, x:x+w]

def crear_directorio(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)  # Crear el directorio si no existe
    else:
        shutil.rmtree(ruta)
        os.makedirs(ruta)  # Limpiar el directorio si ya existe
    return ruta

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

def calcular_centro(bbox):
    x, y, w, h = bbox
    centro_x = x + w // 2
    centro_y = y + h // 2
    return centro_x, centro_y

def umbralizar(image, threshold):
    _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return image

def ensanchar_bbox(bbox, factor):
    x0, y0, w, h = bbox
    x0 -= np.round(w * factor).astype(int)
    y0 -= np.round(h * factor).astype(int)
    w += 2 * np.round(w * factor).astype(int)
    h += 2 * np.round(h * factor).astype(int)
    return np.array([x0, y0, w, h])

def angulo_entre(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    bc = c - b
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angulo = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angulo

def calcular_curvatura_promedio(contorno):
    curvatura_total = 0
    n = len(contorno)
    for i in range(n):
        p1 = contorno[i][0]
        p2 = contorno[(i+1) % n][0]
        p3 = contorno[(i+2) % n][0]
        curvatura = angulo_entre(p1, p2, p3)
        curvatura_total += curvatura
    curvatura_promedio = curvatura_total / n
    return curvatura_promedio

def mostrar_proceso(image, steps, reason=None):
    plt.figure(figsize=(15, 5))
    titles = ['Original', 'Umbralizado', 'Contornos']
    
    for i, (title, step_image) in enumerate(zip(titles, steps)):
        plt.subplot(1, len(steps), i+1)
        plt.imshow(step_image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    if reason:
        plt.suptitle(f'Imagen descartada: {reason}', fontsize=16, color='red')
    else:
        plt.suptitle('Imagen procesada correctamente', fontsize=16, color='green')
    
    plt.show()

def calcular_roughness_index(contorno):
    perimeter_contour = cv2.arcLength(contorno, True)
    hull = cv2.convexHull(contorno)
    perimeter_hull = cv2.arcLength(hull, True)
    roughness_index = perimeter_contour / perimeter_hull
    return roughness_index
    

def filtrar_contornos(image_gray, mostrar):

    if image_gray.size == 0:
        print("La imagen está vacía. Saltando este paso.")
        return False

    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    #Mostrar imagen
    #cv2.imshow('Imagen original', image_gray)
    #cv2.waitKey(0)
    #Mostar histograma
    #plt.hist(image_gray.ravel(), 256, [0, 256])
    #plt.show()
    mean = np.mean(image_gray)
    if(mean < 100):
        #Recortar imagen
        image_gray = recortar_imagen(image_gray, ensanchar_bbox([0, 0, image_gray.shape[1], image_gray.shape[0]], -0.08))
        #umbralizacion dinamica con otsu
        umbral, image_thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        #cv2.imshow('Imagen umbralizada', image_thresh)
        #cv2.waitKey(0)
        #mostrar histograma original con umbral señalado en rojo
        #plt.hist(image_gray.ravel(), 256, [0, 256])
        #plt.axvline(x=umbral, color='r', linestyle='dashed', linewidth=1)
        #plt.show()
        contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    else:
        _, image_thresh = cv2.threshold(image_gray, mean, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_thresh = cv2.cvtColor(image_thresh, cv2.COLOR_GRAY2BGR)
    image_thresh = cv2.cvtColor(image_thresh, cv2.COLOR_BGR2RGB)
    
    steps = [image_gray, image_thresh, image_thresh]
    cv2.drawContours(steps[2], contours, -1, (0, 255, 0), 2)

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    
    if best_cnt is None:
        if mostrar:
            mostrar_proceso(image_gray, steps, reason='No se encontraron contornos válidos')
        return False

    perimeter = cv2.arcLength(best_cnt, True)
    area = cv2.contourArea(best_cnt)
    circularity = 4 * np.pi * area / perimeter**2

    if circularity < 0.6:
        if mostrar:
            mostrar_proceso(image_gray, steps, reason='Contorno no circular')
        return False

    
    if mostrar:
        mostrar_proceso(image_gray, steps)

    return True

def procesar_imagenes(image_l, image_r, input_masks_l, input_masks_r, mostrar):
    masks = []

    # Asegurarse de que ambas listas tengan la misma longitud
    assert len(input_masks_l) == len(input_masks_r), "Las listas de máscaras deben tener la misma longitud"

    # Iterar sobre las máscaras en ambas listas
    for mask_l, mask_r in zip(input_masks_l, input_masks_r):
        bbox_l = mask_l['bbox']
        bbox_r = mask_r['bbox']

        recorte_l = recortar_imagen(image_l, bbox_l)
        recorte_r = recortar_imagen(image_r, bbox_r)

        flag_l = filtrar_contornos(recorte_l, mostrar)
        flag_r = filtrar_contornos(recorte_r, mostrar)

        if flag_l and flag_r:
            masks.append({ 'l': mask_l, 'r': mask_r})

    return masks

# Sufijo de las imágenes a procesar
with open('Images/Pruebas/sufijo.txt', 'r') as f:
    sufijo = f.read().strip()


# Directorios de entrada
image_l = cv2.imread(f'Images/Pruebas/Corrected/cam_sup_l_corrected_{sufijo}.jpg', cv2.IMREAD_GRAYSCALE)
image_r = cv2.imread(f'Images/Pruebas/Corrected/cam_sup_r_corrected_{sufijo}.jpg', cv2.IMREAD_GRAYSCALE)

ruta_masks_l = 'masks/masks_filt_corr_l.json'
ruta_masks_r = 'masks/masks_filt_corr_r.json'

# Máscaras de entrada
masks_l = []
with open(ruta_masks_l, 'r') as archivo:
    masks_l = json.load(archivo)

masks_r = []
with open(ruta_masks_r, 'r') as archivo:
    masks_r = json.load(archivo)

input_boxes_l = []
for mask in masks_l:
    xini = mask['bbox'][0]
    yini = mask['bbox'][1]
    xfin = xini + mask['bbox'][2]
    yfin = yini + mask['bbox'][3]
    input_box = np.array([xini, yini, xfin, yfin])
    input_boxes_l.append(input_box)

input_boxes_r = []
for mask in masks_r:
    xini = mask['bbox'][0]
    yini = mask['bbox'][1]
    xfin = xini + mask['bbox'][2]
    yfin = yini + mask['bbox'][3]
    input_box = np.array([xini, yini, xfin, yfin])
    input_boxes_r.append(input_box)

# Directorios de salida
output_masks_filt_morf = 'masks/masks_filt_morf.json'

# Procesar imágenes de cada directorio
masks_filt_morf = procesar_imagenes(image_l, image_r, masks_l, masks_r, mostrar = False)

# MOstrar número de máscaras filtradas
print(len(masks_filt_morf))

#invertir ensanchamiento de todas las bboxes almacenadas
for mask in masks_filt_morf:
    mask['l']['bbox'] = ensanchar_bbox(mask['l']['bbox'],-0.08)
    mask['r']['bbox'] = ensanchar_bbox(mask['r']['bbox'],-0.08)
# Guardar los datos de las máscaras en un .json
with open(output_masks_filt_morf, 'w') as archivo_json:
    json.dump(masks_filt_morf, archivo_json, default=convertir_a_json, indent=4)

