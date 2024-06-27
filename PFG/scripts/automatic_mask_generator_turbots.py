
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
import os

#This function takes the segmentation parameter of the mask, which is a boolean matrix that is put to true in the coordinates where the mask of the image is located and false inthe rest.
#The mask passed as a parameter is dimensionated to the size of the cuadrant, which is a crop of the original image,so it is necessary to resize it to the original size of the image.
#Depending on the cuadrant number, the coordinates of the mask are adjusted to the original image, don´t forget to also 



def convertir_a_json(objeto):
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()  # Convertir ndarray a lista
    raise TypeError(f"Tipo {type(objeto)} no serializable")

# Definir la función para mostrar las anotaciones
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.array([0,0,1,0.5])
        img[m] = color_mask
    ax.imshow(img)

def remove_overlapped_masks(masks, threshold):
    masks_to_keep = masks.copy()
    indices_to_remove = set()
    mask_sums = [np.sum(mask['segmentation']) for mask in masks_to_keep]

    # Iterar sobre cada par de máscaras
    for i, mask_i in enumerate(masks_to_keep):
        for j, mask_j in enumerate(masks_to_keep[i+1:], start=i+1):
            # Verificar si las máscaras se superponen
            overlap = mask_i['segmentation'] & mask_j['segmentation']

            if np.any(overlap):
                # Calcular el porcentaje de superposición
                overlap_percentage = np.sum(overlap) / mask_sums[i]

                # Si el porcentaje de superposición es mayor que el umbral, marcar para eliminación
                if overlap_percentage > threshold:
                    if mask_sums[i] > mask_sums[j]:
                        indices_to_remove.add(i)
                        
    # Construir lista de máscaras a mantener
    masks_to_keep = [mask for idx, mask in enumerate(masks_to_keep) if idx not in indices_to_remove]

    return masks_to_keep


# Cargar la imagen de ejemplo
image_path = 'Images/Pruebas/cam_sup_l_corrected.jpg'
image_l = cv2.imread(image_path)
image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
alto_imagen_l = image_l.shape[0]
ancho_imagen_l = image_l.shape[1]

image_path = 'Images/Pruebas/cam_sup_r_corrected.jpg'
image_r = cv2.imread(image_path)
image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
alto_imagen_r = image_r.shape[0]
ancho_imagen_r = image_r.shape[1]

#preprocesar imágenes
#convertimos a escala de grises
gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

#reducimos brillo de la imagen
gray_l = cv2.convertScaleAbs(gray_l, alpha=0.5, beta=0)
gray_r = cv2.convertScaleAbs(gray_r, alpha=0.5, beta=0)

#ecualizamos el histograma
gray_l = cv2.equalizeHist(gray_l)
gray_r = cv2.equalizeHist(gray_r)

# Instalar y clonar los repositorios necesarios (esto es solo un ejemplo, no se ejecutará en un script)
# os.system('pip install timm')
# os.system('git clone https://github.com/ChaoningZhang/MobileSAM')

# Importar los módulos necesarios del repositorio clonado
sys.path.append("..")
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# Configurar el modelo SAM
sam_checkpoint = "Modelos/MobileSAM/weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

# Opciones avanzadas de generación de máscaras
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=48,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=15
)


masks_l = []
masks_r = []
masks_l_keep = []
masks_r_keep = []
masks_l = mask_generator.generate(image_l)
masks_r = mask_generator.generate(image_r)

for mask in masks_l: # Añadir el número de cuadrante a cada máscara
    width = mask['bbox'][2]
    height = mask['bbox'][3]
    area = mask['area']
    r1 = width/height
    r2 = height/width
    if  r1 <= 1.2 and r2 <=1.2: #Se filtran las mascaras por la relacion de aspecto de la bounding box
        if area <= 3500: #Se filtran por altura
            masks_l_keep.append(mask)

for mask in masks_r: 
    width = mask['bbox'][2]
    height = mask['bbox'][3]
    area = mask['area']
    r1 = width/height
    r2 = height/width
    if  r1 <= 1.2 and r2 <=1.2: #Se filtran las mascaras por la relacion de aspecto de la bounding box
        if area <= 3500: #Se filtran por altura
            masks_r_keep.append(mask)
    

masks_l_to_keep = remove_overlapped_masks(masks_l_keep, 0.3)
masks_r_to_keep = remove_overlapped_masks(masks_r_keep, 0.3)


# Mostrar las máscaras sobre la imagen
fig, ax = plt.subplots(figsize=(ancho_imagen_l / 100, alto_imagen_l / 100), dpi=100)
ax.imshow(image_l)
show_anns(masks_l_to_keep)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('Images/image_with_masks_left.jpg', dpi=100, bbox_inches='tight', pad_inches=0)
plt.close(fig)

fig, ax = plt.subplots(figsize=(ancho_imagen_r / 100, alto_imagen_r / 100), dpi=100)
ax.imshow(image_r)
show_anns(masks_r_to_keep)
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('Images/image_with_masks_right.jpg', dpi=100, bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Lista de bounding boxes y áreas para las máscaras del lado izquierdo
mask_list_l = []
for mask in masks_l_to_keep:
    mask_list_l.append({
        'bbox': mask['bbox'],
        'area': mask['area'],
    })

# Lista de bounding boxes y áreas para las máscaras del lado derecho
mask_list_r = []
for mask in masks_r_to_keep:
    mask_list_r.append({
        'bbox': mask['bbox'],
        'area': mask['area'],
    })

#Guardar los datos de las máscaras en un .json
with open('masks/masks_l.json', 'w') as archivo_json:
    json.dump(mask_list_l, archivo_json, default=convertir_a_json, indent=4)

with open('masks/masks_r.json', 'w') as archivo_json:
    json.dump(mask_list_r, archivo_json, default=convertir_a_json, indent=4)



