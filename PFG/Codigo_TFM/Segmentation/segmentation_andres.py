import numpy as np
import cv2
import json
import os
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
import math

def cargar_contorno_desde_json(json_data):
    contorno = np.array(json_data['cnt']).reshape(-1, 2)
    return contorno

def dist(p1, p2):
     
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return x0 * x0 + y0 * y0

import numpy as np

def find_closest_points(contour, point1, point2):
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    closest_point1 = None
    closest_distance1 = float('inf')
    
    closest_point2 = None
    closest_distance2 = float('inf')
    
    for point in contour:
        point = tuple(point[0])  # Asegurarse de que el punto esté en la forma correcta
        
        dist1 = euclidean_distance(point1, point)
        dist2 = euclidean_distance(point2, point)
        
        if dist1 < closest_distance1:
            closest_distance1 = dist1
            closest_point1 = point
        
        if dist2 < closest_distance2:
            closest_distance2 = dist2
            closest_point2 = point
    
    return closest_point1, closest_point2, closest_distance1, closest_distance2



def recortar_imagen(imagen, bbox):
    x, y, w, h = bbox
    return imagen[y:y+h, x:x+w]


def get_ext_dist_points(contour):
    # Asegurarse de que el contorno es un arreglo de NumPy
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    
    # Calcular los puntos extremos del contorno
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    vertical_distance = bottommost[1] - topmost[1]  
    horizontal_distance = bottommost[0] - topmost[0]
   
    distance_bottop = math.sqrt(vertical_distance**2 + horizontal_distance**2)

    vertical_distance = rightmost[1] - leftmost[1]
    horizontal_distance = rightmost[0] - leftmost[0]

    distance_rightleft = math.sqrt(vertical_distance**2 + horizontal_distance**2)

    if distance_bottop > distance_rightleft:
        distance = distance_bottop
        axis = 'vertical'
    else:
        distance = distance_rightleft  
        axis = 'horizontal'  
    return distance, axis

def get_ext_points(contour,axis):
        # Asegurarse de que el contorno es un arreglo de NumPy
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
    if axis == 'vertical':
        ext_pts = [topmost, bottommost]
    else:
        ext_pts = [leftmost, rightmost]  
    return ext_pts


def maxDist(points):
 
    n = len(points)
    maxm = 0
    aux = 0

    # Iterate over all possible pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Update maxm
            maxm2 = max(maxm, dist(points[i][0], points[j][0]))
            if maxm2 != maxm:
                maxm = maxm2
                max_len_pts = [[int(points[i][0][0]), int(points[i][0][1])], [int(points[j][0][0]), int(points[j][0][1])]]
 
    # Return actual distance
    if max_len_pts[0][1] > max_len_pts[1][1]:
        aux = max_len_pts[0]
        max_len_pts[0] = max_len_pts[1]
        max_len_pts[1] = aux
    return max_len_pts

def get_centroid(contour):
    # Asegurarse de que el contorno es un arreglo de NumPy
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    
    # Calcular los momentos del contorno
    M = cv2.moments(contour)
    
    # Calcular el centroide si el área del contorno es diferente de cero
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0  # O manejar de otra manera si el contorno es muy pequeño o tiene área cero
    
    return (cX, cY)

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
    

# Paths and constants
files_dir = r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\Corrected'
output_file_path = r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\head_tail_pts.json'
head_tail_pts = {}

# Load contours from JSON file
with open(r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\masks\masks_contours.json', 'r') as archivo:
    masks_contours = json.load(archivo)

#Leer sufijo
with open(r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\sufijo.txt', 'r') as f:
    sufijo = f.read().strip()

# Process each image and find head-tail points
for path_img in os.listdir(files_dir):
    if(path_img == f"cam_sup_l_corrected_{sufijo}.jpg" or path_img == f"cam_sup_r_corrected_{sufijo}.jpg"):
        dict_aux = {}
        idx = 0
        path_img_aux = "/"+path_img
        for mask_contour in masks_contours:
            contorno_l = mask_contour['l']['cnt']
            contorno_r = mask_contour['r']['cnt']
            bbox_l = mask_contour['l']['bbox']
            bbox_r = mask_contour['r']['bbox']
            despx_l = bbox_l[0]
            despy_l = bbox_l[1]
            despx_r = bbox_r[0]
            despy_r = bbox_r[1]
            diff_x = despx_r - despx_l
            diff_y = despy_r - despy_l
            recoter_l = recortar_imagen(cv2.imread(files_dir+f"/cam_sup_l_corrected_{sufijo}.jpg"), bbox_l)
            recoter_r = recortar_imagen(cv2.imread(files_dir+f"/cam_sup_r_corrected_{sufijo}.jpg"), bbox_r)
            max_pts_l = maxDist(contorno_l)
            max_pts_r = maxDist(contorno_r)
            max_dist_l = math.sqrt((max_pts_l[0][0] - max_pts_l[1][0])**2 + (max_pts_l[0][1] - max_pts_l[1][1])**2)
            max_dist_r = math.sqrt((max_pts_r[0][0] - max_pts_r[1][0])**2 + (max_pts_r[0][1] - max_pts_r[1][1])**2)
            max_dist = max(max_dist_l, max_dist_r)
            if(max_dist == max_dist_l):
                ext_pts_l = max_pts_l
                #los puntos extremos en r se calculan en base a la diferencia entre la bbox de r y la bbox de l
                ext_pts_r_aux = [[max_pts_l[0][0] + diff_x, max_pts_l[0][1] + diff_y], [max_pts_l[1][0] + diff_x, max_pts_l[1][1] + diff_y]]
                #Se buscan los puntos del contorno r más cercanos a los puntos trasladados de l
                ext_pt_r_1, ext_pt_r_2, distance_r_1, ditance_r_2 = find_closest_points(contorno_r, ext_pts_r_aux[0], ext_pts_r_aux[1])
                if distance_r_1 > 7 or ditance_r_2 > 7:
                    continue
                ext_pts_r = [ext_pt_r_1, ext_pt_r_2]
                #buscar el par de puntos extremos en r que esten más cerca de ext_pts_r_aux
                

            else:
                ext_pts_r = max_pts_r
                #los puntos extremos en l se calculan en base a la diferencia entre la bbox de l y la bbox de r
                ext_pts_l_aux = [[max_pts_r[0][0] - diff_x, max_pts_r[0][1] - diff_y], [max_pts_r[1][0] - diff_x, max_pts_r[1][1] - diff_y]]
                ext_pt_l_1, ext_pt_l_2, distance_l_1, ditance_l_2 = find_closest_points(contorno_l, ext_pts_l_aux[0], ext_pts_l_aux[1])
                if distance_l_1 > 5 or ditance_l_2 > 5:
                    continue
                ext_pts_l = [ext_pt_l_1, ext_pt_l_2]
                
            if 'cam_sup_l' in path_img:
                cnt = contorno_l
                centroid = get_centroid(cnt)
                dict_aux[idx] = [centroid, [list(pt) for pt in ext_pts_l]]
                idx += 1
                head_tail_pts[path_img_aux] = dict_aux
            elif 'cam_sup_r' in path_img:
                cnt = contorno_r
                centroid = get_centroid(cnt)
                dict_aux[idx] = [centroid, [list(pt) for pt in ext_pts_r]]
                idx += 1
                head_tail_pts[path_img_aux] = dict_aux

# Save head-tail points to JSON
with open(output_file_path, "w") as outfile:
    json.dump(head_tail_pts, outfile, default=convertir_a_json, indent=4)

print("Puntos head-tail guardados en el archivo JSON:", output_file_path)



