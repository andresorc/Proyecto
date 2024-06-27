from reolinkapi import Camera
import numpy as np
import cv2
import csv

id = 'prueba'

ip_sup_d = "192.168.1.1"
ip_sup_i = "192.168.1.2"
#ip_lat = "138.100.58.188"
passw = "CITSEM_pescanova"

cam_sup_d = Camera(ip_sup_d, password = passw)
cam_sup_i = Camera(ip_sup_i, password = passw)
#cam_lat = Camera(ip_lat, password = passw)

print('Sincronizando')
cam_sup_d.setDateTime()
cam_sup_i.setDateTime()
print('Fin sincronizacion')

print('Tomando imagen derecha')
img_sup_d = np.array(cam_sup_d.get_snap())
print('Tomando imagen izquierda')
img_sup_i = np.array(cam_sup_i.get_snap())
#img_lat = np.array(cam_lat.get_snap())

img_sup_d = cv2.cvtColor(img_sup_d, cv2.COLOR_RGB2BGR)
img_sup_i = cv2.cvtColor(img_sup_i, cv2.COLOR_RGB2BGR)
#img_lat = cv2.cvtColor(img_lat, cv2.COLOR_RGB2BGR)

cv2.imwrite(f'Calibration/Stereo/Images/Prueba/cam_sup_r_fig_{id}.jpg', img_sup_d)
cv2.imwrite(f'Calibration/Stereo/Images/Prueba/cam_sup_l_fig_{id}.jpg', img_sup_i)
#cv2.imwrite(f'Depth_estimation/Images/cam_lat_fig_{id}.jpg', img_lat)

if  len(img_sup_d) > 0:
    print("Imagen derecha guardada")

if len(img_sup_i) > 0:
    print("Imagen izquierda guardada")

