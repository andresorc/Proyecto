import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
from collections import Counter
import math
import os


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

def ensanchar_bbox(bbox, factor):
    x0, y0, w, h = bbox
    x0 -= np.round(w * factor).astype(int)
    y0 -= np.round(h * factor).astype(int)
    w += 2 * np.round(w * factor).astype(int)
    h += 2 * np.round(h * factor).astype(int)
    return np.array([x0, y0, w, h])


# Generar la máscara de un objeto en una imagen dada una bounding box
def generar_mascara(imagen, bbox):
    x, y, w, h = bbox['bbox']
    return imagen[y:y + h, x:x + w]

# Mostrar las imágenes y las bounding boxes para verificación
def mostrar_imagen_con_bounding_boxes(imagen, bounding_boxes, titulo):
    fig, ax = plt.subplots(1)
    ax.imshow(imagen)
    for bbox in bounding_boxes:
        x, y, w, h = bbox['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title(titulo)
    plt.show()

# Definir región de búsqueda en el segundo frame basado en la bounding box del primer frame
def definir_region_de_busqueda(bbox, margen_x=400, margen_y=400, img_shape=None):
    x, y, w, h = bbox['bbox']
    x_min = max(x - margen_x, 0)
    y_min = max(y - margen_y, 0)
    x_max = min(x + w + margen_x, img_shape[1] if img_shape else x + w + margen_x)
    y_max = min(y + h + margen_y, img_shape[0] if img_shape else y + h + margen_y)
    return x_min, y_min, x_max, y_max

# Calcular la correlación cruzada entre dos máscaras
def correlate_masks(mascara1, mascara2):
    mascara1 = mascara1.astype('float32')
    mascara2 = mascara2.astype('float32')
    if(mascara2.shape[0] == 0 or mascara2.shape[1] == 0):
        return None, None
    correlacion = cv2.matchTemplate(mascara1, mascara2, method=cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlacion)
    return max_loc,max_val

def calcular_centro(bbox):
    x, y, w, h = bbox['bbox']
    centro_x = x + w // 2
    centro_y = y + h // 2
    return centro_x, centro_y

# Función para calcular el ángulo entre dos puntos en grados
def calcular_angulo(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angulo if angulo >= 0 else angulo + 360

def calcular_distancia(punto1, punto2):
    return np.sqrt((punto1[0] - punto2[0]) ** 2 + (punto1[1] - punto2[1]) ** 2)

# Función para calcular la pendiente entre dos puntos
def calcular_pendiente(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    if x2 != x1:
        return (y2 - y1) / (x2 - x1)
    else:
        return float('inf')  # Pendiente infinita si x2 == x1


def dibujar_lineas_correspondencia_filtradas(imagen1, imagen2, correspondencias):
    angulos = [calcular_angulo(corr[0], corr[1]) for corr in correspondencias]
    correlaciones = [corr[2] for corr in correspondencias]
    diferencias_pixeles = [calcular_distancia(corr[0], corr[1]) for corr in correspondencias]

    # Mostrar todas las correspondencias
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(imagen1)
    axs[1].imshow(imagen2)

    for corr in correspondencias:
        color = tuple(np.random.rand(3))
        cx1, cy1 = corr[0]
        cx2, cy2 = corr[1]

        radius = 5
        circle1 = patches.Circle((cx1, cy1), radius, color=color, fill=True)
        circle2 = patches.Circle((cx2, cy2), radius, color=color, fill=True)
        axs[0].add_patch(circle1)
        axs[1].add_patch(circle2)

        con = patches.ConnectionPatch(xyA=(cx2, cy2), xyB=(cx1, cy1),
                                      coordsA="data", coordsB="data",
                                      axesA=axs[1], axesB=axs[0], color=color)
        axs[1].add_artist(con)

    plt.show()

    # Filtrar por ángulo
    angulo_promedio = Counter(angulos).most_common(1)[0][0]
    margen_angulo = 20  # Por ejemplo, 20 grados de margen
    angulos_filtrados = [(corr, ang) for corr, ang in zip(correspondencias, angulos) if abs(ang - angulo_promedio) < margen_angulo]

    # Filtrar por correlación
    umbral_correlacion = 0.95
    correspondencias_filtradas_por_correlacion = [(corr, ang, diff) for corr, ang, diff, cor in zip(correspondencias, angulos, diferencias_pixeles, correlaciones) if cor >= umbral_correlacion]

    if not correspondencias_filtradas_por_correlacion:
        return None, None, None, None

    # Extraer las nuevas listas después del filtrado por correlación
    correspondencias, angulos, diferencias_pixeles = zip(*correspondencias_filtradas_por_correlacion)

    # Filtrar por distancia promedio
    diferencia_promedio = np.mean(diferencias_pixeles)
    margen_distancia = 30  # Por ejemplo, 30 píxeles de margen
    correspondencias_filtradas_por_distancia = [(corr, ang) for corr, ang, diff in zip(correspondencias, angulos, diferencias_pixeles) if abs(diff - diferencia_promedio) < margen_distancia]

    if not correspondencias_filtradas_por_distancia:
        return None, None, None, None

    # Extraer las nuevas listas después del filtrado por distancia
    correspondencias, angulos = zip(*correspondencias_filtradas_por_distancia)

    # Dibujar las líneas filtradas y calcular la diferencia promedio en píxeles y pendiente promedio
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(imagen1)
    axs[1].imshow(imagen2)

    diferencias_pixeles_final = []
    pendientes = []

    for corr, _ in angulos_filtrados:
        color = tuple(np.random.rand(3))
        cx1, cy1 = corr[0]
        cx2, cy2 = corr[1]

        radius = 5
        circle1 = patches.Circle((cx1, cy1), radius, color=color, fill=True)
        circle2 = patches.Circle((cx2, cy2), radius, color=color, fill=True)
        axs[0].add_patch(circle1)
        axs[1].add_patch(circle2)

        con = patches.ConnectionPatch(xyA=(cx2, cy2), xyB=(cx1, cy1),
                                      coordsA="data", coordsB="data",
                                      axesA=axs[1], axesB=axs[0], color=color)
        axs[1].add_artist(con)

        diferencias_pixeles_final.append(calcular_distancia((cx1, cy1), (cx2, cy2)))
        pendientes.append(calcular_pendiente((cx1, cy1), (cx2, cy2)))

    plt.show()

    diferencia_promedio_final = np.mean(diferencias_pixeles_final) if diferencias_pixeles_final else None
    pendiente_promedio = np.mean(pendientes) if pendientes else None

    correspondencias_filtradas = [corr for corr, _ in angulos_filtrados]

    return diferencia_promedio_final, pendiente_promedio, np.mean(correlaciones), correspondencias_filtradas

# Rutas a los archivos JSON e imágenes
# Solicitar al usuario el sufijo para modificar la ruta de los archivos
sufijo = input("Por favor, introduce el sufijo para los archivos (ejemplo: '1', '2'): ")

# Formar la ruta del archivo JSON para bounding boxes usando el sufijo proporcionado
ruta_bounding_boxes = f'masks/masks_{sufijo}.json'

# Guardar el sufijo en un archivo de texto
with open('Images/Pruebas/sufijo.txt', 'w') as f:
    f.write(sufijo)

# Intentar cargar el archivo JSON con bounding boxes
try:
    with open(ruta_bounding_boxes, 'r') as archivo:
        bounding_boxes = json.load(archivo)
    print("Bounding boxes cargados con éxito.")
except FileNotFoundError:
    print(f"No se encontró el archivo: {ruta_bounding_boxes}")
except json.JSONDecodeError:
    print("Error al decodificar el archivo JSON.")

# Directorio de imágenes modificando la ruta basado en el sufijo ingresado
ruta_imagen_frame1 = f'Images/Pruebas/Corrected/cam_sup_l_corrected_{sufijo}.jpg'
ruta_imagen_frame2 = f'Images/Pruebas/Corrected/cam_sup_r_corrected_{sufijo}.jpg'


# Cargar las imágenes
imagen_frame1 = cv2.imread(ruta_imagen_frame1)
imagen_frame2 = cv2.imread(ruta_imagen_frame2)

# Convertir a escala de grises para procesamiento
imagen_frame1_gray = cv2.cvtColor(imagen_frame1, cv2.COLOR_BGR2GRAY)
imagen_frame2_gray = cv2.cvtColor(imagen_frame2, cv2.COLOR_BGR2GRAY)


# Convertir a RGB para representacion
imagen_frame1_rgb = cv2.cvtColor(imagen_frame1, cv2.COLOR_BGR2RGB)
imagen_frame2_rgb = cv2.cvtColor(imagen_frame2, cv2.COLOR_BGR2RGB)

# Procesar cada objeto del primer frame y encontrar la mejor correspondencia en el segundo frame
correspondencias = []
for bbox1 in bounding_boxes:
    bbox1['bbox'] = ensanchar_bbox(bbox1['bbox'], 0.08)
    _, y, _, _ = bbox1['bbox']
    if(y > 1350 ):
        continue
    mascara = generar_mascara(imagen_frame1_gray, bbox1)
    mascara_original = mascara

    xmin, ymin, xmax, ymax = definir_region_de_busqueda(bbox1, img_shape=imagen_frame2_gray.shape)
    region = imagen_frame2_gray[ymin:ymax, xmin:xmax]
    #mostrar mascara
    #plt.imshow(mascara, cmap='gray')
    #plt.show()
    #mostrar region
    #plt.imshow(region, cmap='gray')
    #plt.show()
    
    
    if region.shape[0] * region.shape[1] < mascara.shape[0] * mascara.shape[1]:
        continue
    
    max_loc, max_val = correlate_masks(region, mascara)
    if(max_val == None):
        continue
    centro2 = (max_loc[0] + xmin + mascara.shape[1] // 2, max_loc[1] + ymin + mascara.shape[0] // 2)
    centro1 = calcular_centro(bbox1)
    # Agregar las correspondencias y los valores de correlación a la lista
    correspondencias.append((centro1, centro2, max_val))


# Dibujar las líneas de correspondencia y calcular la diferencia promedio en píxeles y la pendiente promedio
diferencia_promedio, pendiente_promedio, correlacion_promedio, correspondencias = dibujar_lineas_correspondencia_filtradas(imagen_frame1_rgb, imagen_frame2_rgb, correspondencias)
#sacar numero de corresondencias filtradas 
num_correspondencias = len(correspondencias)
print("Número de correspondencias filtradas:", num_correspondencias)
#Se calcula la diferencia promedio y la pendiente promedio otra vez, pero solo sobre las correspondencias filtradas
#para evitar que se tomen en cuenta las correspondencias que no se han filtrado
diferencia_promedio = np.mean([calcular_distancia(corr[0], corr[1]) for corr in correspondencias]) if correspondencias else None
pendiente_promedio = np.mean([calcular_pendiente(corr[0], corr[1]) for corr in correspondencias]) if correspondencias else None
print("Diferencia promedio:", diferencia_promedio)
print("Pendiente promedio:", pendiente_promedio)
print("Correlación promedio:", correlacion_promedio)

#Mostrar 5 mejores correpondencias en las imagenes
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(imagen_frame1_rgb)
axs[1].imshow(imagen_frame2_rgb)

for corr in correspondencias[:5]:
    color = tuple(np.random.rand(3))
    cx1, cy1 = corr[0]
    cx2, cy2 = corr[1]

    radius = 5
    circle1 = patches.Circle((cx1, cy1), radius, color=color, fill=True)
    circle2 = patches.Circle((cx2, cy2), radius, color=color, fill=True)
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)

    con = patches.ConnectionPatch(xyA=(cx2, cy2), xyB=(cx1, cy1),
                                  coordsA="data", coordsB="data",
                                  axesA=axs[1], axesB=axs[0], color=color)
    axs[1].add_artist(con)

plt.show()

# Filtrar las máscaras originales según las correspondencias filtradas
masks_l_filtradas = []
masks_r_filtradas = []

for corr_filtrada in correspondencias:
    centro1, centro2, correlacion_max = corr_filtrada
    # Encontrar la máscara correspondiente al centroide en la imagen izquierda
    for bbox1 in bounding_boxes:
        if calcular_centro(bbox1) == centro1:
            bbox1['centroide'] = centro1
            x, y, w, h = bbox1['bbox']
            masks_l_filtradas.append(bbox1)
            mask_r = {}
            mask_r['bbox'] = [centro2[0] - w // 2, centro2[1] - h // 2, w, h]
            mask_r['centroide'] = centro2
            masks_r_filtradas.append(mask_r)
            break

# Guardar las máscaras filtradas con los centroides en archivos JSON
with open('masks/masks_filt_corr_l.json', 'w') as archivo:
    json.dump(masks_l_filtradas, archivo, default = convertir_a_json, indent=4)

with open('masks/masks_filt_corr_r.json', 'w') as archivo:
    json.dump(masks_r_filtradas, archivo, default = convertir_a_json, indent=4)

# Guardar datos de pendiente y diferencia promedio en un archivo JSON
datos = {'diferencia_promedio': diferencia_promedio, 'pendiente_promedio': pendiente_promedio}
with open('masks/datos_correspondencias.json', 'w') as archivo:
    json.dump(datos, archivo, default = convertir_a_json, indent=4)

