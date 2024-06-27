import cv2
import os
import math
import numpy as np
import json

def get_extent(cnt, area):

    _,_,w,h = cv2.boundingRect(cnt)
    extent = float(area)/(w*h)

    return extent

def get_eccentricity(cnt):

    _,(ma,MA),_ = cv2.fitEllipse(cnt)
    eccentricity = np.sqrt(1-(ma/MA)**2)

    return eccentricity

def get_solidity(cnt, area):

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    return solidity

def get_intensity(cnt, bin_img, img):

    mask = np.zeros(bin_img.shape, np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    mean = cv2.mean(img ,mask = mask)
    
    return mean

def get_centroid(cnt):

    M = cv2.moments(cnt)
    cx = 0
    cy = 0
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    
    return [cx,cy]

def get_extreme_points(cnt):

    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    return [topmost, bottommost, leftmost, rightmost]

def dist(p1, p2):
     
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return x0 * x0 + y0 * y0


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


imgs_dir = "Depth_estimation/Images/Undistorted/Corrected/"
files_dir = 'Segmentation/'
name_fich_carac_morf_text = "caracteristicas_morfologicas"
path_ref_img_r = "cam_sup_r_fig_ref_corrected.jpg"
path_ref_img_l = "cam_sup_l_fig_ref_corrected.jpg"
brown = [30, 21, 30]

draw = 0
guardar_carac_morfologicas = 1
n_real = 0
head_tail_pts = {}
dict_aux = {}

ref_img_r = cv2.imread(os.path.join(imgs_dir,path_ref_img_r))
ref_img_l = cv2.imread(os.path.join(imgs_dir,path_ref_img_l))

for path_img in os.listdir(imgs_dir):

    #or 'cam_sup_l_fig_21' in path_img or 'cam_sup_l_fig_22' in path_img
    if path_img.endswith('.jpg') and ('cam_sup_l' in path_img or 'cam_sup_r' in path_img) and path_img != path_ref_img_r and path_img != path_ref_img_l:
    #if path_img.endswith('.jpg') and ('cam_sup_r_fig_14' in path_img ) and path_img != path_ref_img_r and path_img != path_ref_img_l and path_img != path_ref_img_lat:
        if 'cam_sup_r' in path_img:

            ref_img = ref_img_r

        else:
            
            ref_img = ref_img_l

        fich_carac_morf_abierto = True

        #Intento abrir el fichero donde quiero intentar guardar las características morfológicas de los objetos segmentados
        try:
            file_morf = open(f"{files_dir}{name_fich_carac_morf_text}_{path_img.split('.', 1)[0]}.txt", "w")
        except IOError:
            print(f"El fichero {name_fich_carac_morf_text}_{path_img.split('.', 1)[0]}.txt no se ha podido abrir")
            fich_carac_morf_abierto = False
        
        img = cv2.imread(os.path.join(imgs_dir,path_img))

        no_back_img = cv2.subtract(img, ref_img)

        no_brown_img = no_back_img.copy()
        no_brown_img[:,:,0] = cv2.subtract(no_brown_img[:,:,0], brown[0])
        no_brown_img[:,:,1] = cv2.subtract(no_brown_img[:,:,1], brown[1])
        no_brown_img[:,:,2] = cv2.subtract(no_brown_img[:,:,2], brown[2])

        enh_img = cv2.addWeighted(no_brown_img, 3, np.zeros(no_brown_img.shape, no_brown_img.dtype), 0,  80)

        if (draw):
            cv2.imshow("Imagen original", img)
            cv2.waitKey(500)

            cv2.imshow("Imagen sin fondo", no_back_img)
            cv2.waitKey(500)

            cv2.imshow("Imagen con contraste", no_brown_img)
            cv2.waitKey(500)

            cv2.imshow("Imagen sin peso", enh_img)
            cv2.waitKey(0)
    
        #Pasar la imagen a escala de grises
        img1 = cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)

        if (draw):
            cv2.imshow("Imagen en escala de grises", img1)
            cv2.waitKey(500)

        #Calculo el tamaño de la imagen (ancho*alto)
        #im_size = img.shape[0]*img.shape[1]
        #Calculo el tamaño de bloque correspondiente al que calcula Matlab por defecto
        #blockSize = 2*(math.floor(im_size/16))+1
        #cv.adaptiveThreshold(	src, maxValue, adaptiveMethod, thresholdType, blockSize, C) ->	dst
        #C: Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as well.
        #No sé exactamente qué es C, el resto de los valores los tengo controlados
        #No me deja usar el blockSize calculado (creo que en Matlab se coge por defecto en valor de blockSize que yo he calculado)
        #Dejo el valor 199 porque es el que he visto en internet
        bw = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 1.05)

        kernel = np.ones((5,5), np.uint8)

        ## OBTENER LOS BORDES Y OBJETOS DE LA IMAGEN CON SUS CARACTERÍSTICAS
        #Detecto los bordes

        if (draw):
            cv2.imshow("Imagen binarizada", bw)
            cv2.waitKey(0)

        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #Los pinto en la imagen
        img_final = img.copy()
        cv2.drawContours(img_final, contours, -1, (255,0,0))

        if (draw):
            cv2.imshow("Imagen con contornos", img_final)
            cv2.waitKey(500)
                
        #EXTRAER Y GUARDAR LAS CARACTERÍSTICAS MNORFOLÓGICAS DE LOS OBJETOS DETECTADOS
        #Vecindad de 8
        #Obtengo array etiquetado y número de objetos en la imagen
        if (guardar_carac_morfologicas):

            #Obtengo las características morfológicas de los objetos
            for cnt in contours:

                area = cv2.contourArea(cnt)

                if area > 5:

                    solidity = get_solidity(cnt, area)
                    extent = get_extent(cnt, area)
                    eccentricity = get_eccentricity(cnt)
                    perimeter = cv2.arcLength(cnt,True)
                    intensity = get_intensity(cnt, bw, img)

                else:

                    solidity = 0
                    extent = 0
                    eccentricity = 0

                if area > 1000 and area <13000 and solidity > 0.81 and solidity < 0.95 and extent > 0.5 and extent < 0.8 and eccentricity > 0.2 and eccentricity < 0.82:
                    
                    centroid = get_centroid(cnt)

                    #print(solidity)

                    if max(abs(intensity[0]-intensity[1]),abs(intensity[0]-intensity[2])) < 15:
                        color = 'blanco'
                    elif np.max(intensity) == intensity[1]:
                        color = 'verde'
                    else:
                        color = 'rojo'

                    n_real = n_real + 1
                    file_morf.write(f"Object {n_real} area: {area} ")
                    file_morf.write(f"Object {n_real} perimeter: {perimeter} ")
                    file_morf.write(f"Object {n_real} centroid: {centroid} ")
                    file_morf.write(f"Object {n_real} extent: {extent} ")
                    file_morf.write(f"Object {n_real} eccentricity: {eccentricity} ")
                    file_morf.write(f"Object {n_real} color: {color} ")
                    file_morf.write(f"Object {n_real} mean intensity: {intensity[0]} ")
                    file_morf.write(f"Object {n_real} solidity: {solidity}\n")

                    ext_pts = maxDist(cnt)

                    img_final = cv2.circle(img_final, (math.floor(centroid[0]), math.floor(centroid[1])), radius=20, color=(255, 255, 255), thickness=-1)
                    img_final = cv2.putText(img_final, f"{str(n_real)} {color}", (math.floor(centroid[0])-10, math.floor(centroid[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                    
                    img_final = cv2.circle(img_final, (ext_pts[0]), radius=5, color=(255, 0, 0), thickness=-1)
                    img_final = cv2.circle(img_final, (ext_pts[1]), radius=5, color=(255, 0, 0), thickness=-1)

                    print("Analizado")

                    #dict_aux[n_real] = [centroid, ext_pts, color]

            cv2.imwrite(f'{files_dir}segmentacion_{path_img.rpartition("_")[0]}.jpg', img_final)
            head_tail_pts[path_img] = dict_aux

            n_real = 0
            dict_aux = {}

        if (fich_carac_morf_abierto):
            file_morf.close

json_object = json.dumps(head_tail_pts, indent=4)
 
# Writing to sample.json
with open(f"{files_dir}head_tail_pts.json", "w") as outfile:
    outfile.write(json_object)