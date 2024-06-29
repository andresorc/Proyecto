import cv2
import numpy as np
import os
import json
import csv

dir_mx_d = 'Calibration/Cam_sup_d'
dir_mx_i = 'Calibration/Cam_sup_i'
dir_stereo = 'Calibration/Stereo'
#imgs_dir = 'Depth_estimation/Images'
imgs_dir = r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas'
imgs_dir_corrected = r'C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\corrected'
dir_json = 'Calibration/Stereo/params.json'

mtx_cam_r = np.loadtxt(os.path.join(dir_mx_d, 'mtx_cam_sup_d.csv'), delimiter=',')
dist_cam_r = np.loadtxt(os.path.join(dir_mx_d, 'dist_cam_sup_d.csv'), delimiter=',')

mtx_cam_l = np.loadtxt(os.path.join(dir_mx_i, 'mtx_cam_sup_i.csv'), delimiter=',')
dist_cam_l = np.loadtxt(os.path.join(dir_mx_i, 'dist_cam_sup_i.csv'), delimiter=',')

R = np.loadtxt(os.path.join(dir_stereo, 'R_stereo.csv'), delimiter=',')
T =  np.loadtxt(os.path.join(dir_stereo, 'T_stereo.csv'), delimiter=',')

#Recorrer imgs_dir_corrected para determinar el sufijo de la imagen que se va a procesar en función de la imagen con el número de sufijo más alto
#Las imágenes tienen el siguiente formato: cam_sup_l_corrected_sufijo.jpg, cam_sup_r_corrected_sufijo.jpg
#El sufijo es un número entero que se incrementa en 1 para cada imagen
#Se obtiene el sufijo de la imagen con el número más alto y se incrementa en 1 para obtener el sufijo de la siguiente imagen

sufijo = 0
remove = []
imgs_name = os.listdir(imgs_dir_corrected)
for img_name in imgs_name:
    if not img_name.endswith('.jpg'):
        remove.append(img_name)

for item in remove:
    imgs_name.remove(item)

for img_name in imgs_name:
    if 'cam_sup_l_corrected_' in img_name:
        sufijo = max(sufijo, int(img_name.split('_')[-1].split('.')[0]))

sufijo += 1


imgs_name = os.listdir(imgs_dir)
for img_name in imgs_name:
    if not img_name.endswith('.jpg'):
        remove.append(img_name)

for item in remove:
    imgs_name.remove(item)

imageSize = cv2.imread(os.path.join(imgs_dir,imgs_name[0])).shape[0:2]

imageSize = np.flip(np.asarray(imageSize))

R1, R2, new_mtx_cam1, new_mtx_cam2, Q, _, _ = cv2.stereoRectify(mtx_cam_l, dist_cam_l, mtx_cam_r, dist_cam_r, imageSize, R, T, alpha=0)

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_cam_l, dist_cam_l, R1, new_mtx_cam1, imageSize, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_cam_r, dist_cam_r, R2, new_mtx_cam2, imageSize, cv2.CV_32FC1)

imgs_name = sorted(imgs_name)
cam_sup_i_images_names = imgs_name[len(imgs_name)//3:2*len(imgs_name)//3]
cam_sup_d_images_names = imgs_name[2*len(imgs_name)//3:]

for (img_l_name,img_r_name) in zip(cam_sup_i_images_names,cam_sup_d_images_names):

    print(f"{img_l_name},{img_r_name}")

    img_r = cv2.imread(os.path.join(imgs_dir, img_r_name))
    img_l = cv2.imread(os.path.join(imgs_dir, img_l_name))

    rect_img_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rect_img_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    cv2.imwrite(f'{imgs_dir}/Corrected/{img_l_name.rpartition(".")[0]}_corrected_{sufijo}.jpg', rect_img_l)
    cv2.imwrite(f'{imgs_dir}/Corrected/{img_r_name.rpartition(".")[0]}_corrected_{sufijo}.jpg', rect_img_r)

print(Q)

json_dict = {}
json_dict['focal_length'] = Q[2][3]
json_dict['baseline'] = 1/Q[3][2]
json_dict['opt_cnt_x'] = -Q[0][3]
json_dict['opt_cnt_y'] = -Q[1][3]

json_object = json.dumps(json_dict, indent=4)
 
# Writing to sample.json
with open(f"{dir_json}", "w") as outfile:
    outfile.write(json_object)