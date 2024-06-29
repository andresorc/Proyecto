
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

import numpy as np

def remove_overlapped_masks(masks, threshold):
    masks_to_keep = masks.copy()
    indices_to_remove = set()
    mask_areas = [np.sum(mask['segmentation']) for mask in masks_to_keep]

    num_masks = len(masks_to_keep)
    
    # Iterar sobre cada par de máscaras
    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            mask_i = masks_to_keep[i]
            mask_j = masks_to_keep[j]

            # Verificar si las máscaras se superponen
            overlap = mask_i['segmentation'] & mask_j['segmentation']

            if np.any(overlap):
                # Calcular el porcentaje de superposición
                overlap_percentage_i = np.sum(overlap) / mask_areas[i]
                overlap_percentage_j = np.sum(overlap) / mask_areas[j]

                # Si el porcentaje de superposición es mayor que el umbral, decidir cuál máscara mantener
                if overlap_percentage_i > threshold or overlap_percentage_j > threshold:
                    if mask_areas[i] < mask_areas[j]:
                        indices_to_remove.add(j)
                    else:
                        indices_to_remove.add(i)
    
    # Construir lista de máscaras a mantener
    masks_to_keep = [mask for idx, mask in enumerate(masks_to_keep) if idx not in indices_to_remove]

    return masks_to_keep



# Se leen las máscaras de la lista

#Se pide el sufijo de la imagen 
sufijo = input('Introduce el sufijo de la imagen: ')

# Cargar la imagen de ejemplo
image_path = f'Images/Pruebas/cam_sup_l_corrected_{sufijo}.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Instalar y clonar los repositorios necesarios (esto es solo para colab, no se ejecutará en un script)
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
    points_per_side=64,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.92,
    crop_n_layers=2,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=15
)


masks = mask_generator.generate(image)
masks_to_keep = []

for mask in masks: # Añadir el número de cuadrante a cada máscara
    area = mask['area']
    if area <= 4800: #Se filtran por altura
        masks_to_keep.append(mask)



masks_to_keep = remove_overlapped_masks(masks_to_keep, 0.2)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

# Lista de bounding boxes y áreas para las máscaras del lado izquierdo
mask_list = []
for mask in masks_to_keep:
    mask_list.append({
        'bbox': mask['bbox'],
    })

#Guardar los datos de las máscaras en un .json
with open(f'masks/masks_{sufijo}.json', 'w') as archivo_json:
    json.dump(mask_list, archivo_json, default=convertir_a_json, indent=4)




