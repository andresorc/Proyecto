import json
import csv
import math
import cv2
import numpy as np

def add_text_box(image, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_type = cv2.LINE_AA

    # Obtener el tamaño del texto
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Coordenadas del cuadro de texto
    text_origin = (pos[0] - text_size[0], pos[1])
    text_bottom_left = (text_origin[0], text_origin[1] + text_size[1] + 10)

    # Crear un fondo negro para el texto
    cv2.rectangle(image, text_origin, (image.shape[1],image.shape[0]), (0, 0, 0), cv2.FILLED)

    # Escribir el texto en rojo
    cv2.putText(image, text, (text_origin[0] + 5, text_origin[1] + text_size[1] + 5),
                font, font_scale, (0, 0, 255), font_thickness, line_type)

json_pts_dir = r"C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\head_tail_pts.json"
json_params_dir = 'Codigo_TFM/Calibration/Stereo/params.json'
depth_csv_dir = 'Codigo_TFM/Depth_estimation/depth.csv'
imgs_dir = r"C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Pruebas\Corrected"
save_imgs_dir = r"C:\Users\andre\OneDrive\Escritorio\UNI\TFG\Codigo\Proyecto\PFG\Images\Results"
lengths = []
draw_length = 1
length_threshold_cm = 6  # Longitud máxima en cm


with open(json_pts_dir) as json_pts_file:
    pts_dict = json.load(json_pts_file)

with open(json_params_dir) as json_params_file:
    params_dict = json.load(json_params_file)

with open(depth_csv_dir, 'w', newline='') as depth_csv:
    writer = csv.writer(depth_csv, delimiter=';')
    focal_length = params_dict['focal_length']
    baseline = params_dict['baseline']
    opt_cnt_x = params_dict['opt_cnt_x']
    opt_cnt_y = params_dict['opt_cnt_y']

    writer.writerow(["imagen", "id_pez", "profundidad de cabeza", "profundidad cuerpo", "profundidad cola", "longitud proyectada estimada (px)", "longitud proyectada estimada (m)", "longitud real estimada (m)", "longitud real (m)", "error absoluto (mm)"])

    for l_img_name in pts_dict.keys():
        if '_l_' in l_img_name:
            r_img_name = l_img_name.replace('_l_', '_r_')
            obj_counter = 1  # Contador de objetos

            for id in pts_dict[l_img_name].keys():
                pts_l = pts_dict[l_img_name][id]
                centroid_l = pts_l[0]
                head_l = pts_l[1][0]
                tail_l = pts_l[1][1]

                pts_r = pts_dict[r_img_name][id]
                centroid_r = pts_r[0]
                head_r = pts_r[1][0]
                tail_r = pts_r[1][1]

                real_length = 0.04  # Longitud real por defecto

                head_depth = baseline * focal_length / abs(head_l[0] - head_r[0])
                if centroid_l[0] == centroid_r[0]:
                    # Handle the case where there is no horizontal displacement
                    # For example, set body_depth to None or a default value
                    continue
                else:
                    body_depth = baseline * focal_length / abs(centroid_l[0] - centroid_r[0])
                tail_depth = baseline * focal_length / abs(tail_l[0] - tail_r[0])
                proyected_length = np.linalg.norm(np.asarray(head_l) - np.asarray(tail_l))
                proyected_length_m = body_depth * proyected_length / focal_length
                real_estimated_length_m = math.sqrt(abs(head_depth - tail_depth)**2 + proyected_length_m**2)

                if real_estimated_length_m * 100 <= length_threshold_cm:  # Filtra longitudes mayores a 6 cm
                    writer.writerow([l_img_name.partition("_l_")[2], id, head_depth, body_depth, tail_depth, proyected_length, proyected_length_m, real_estimated_length_m, real_length, abs(real_estimated_length_m - real_length) * 1000])
                    print(f'{l_img_name.partition("_l_")[2]}/{id} -> cabeza {head_l[1] - head_r[1]} / cuerpo {centroid_l[1] - centroid_r[1]} / cola {tail_l[1] - tail_r[1]}')
                    lengths.append([centroid_r, centroid_l, head_r, head_l, tail_r, tail_l, real_estimated_length_m, obj_counter])

                    if draw_length:
                        img_r = cv2.imread(f"{imgs_dir}/{r_img_name}")
                        img_l = cv2.imread(f"{imgs_dir}/{l_img_name}")

                        for item in lengths:

                                # Color para el número del objeto
                                number_color = (0, 0, 255)  # Rojo
                                # Color para la medida del objeto
                                measurement_color = (0, 255, 255)  # Amarillo

                                img_r = cv2.putText(img_r, f"{item[7]}", (item[0][0] - 50, item[0][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, number_color, 2, cv2.LINE_AA)
                                img_r = cv2.putText(img_r, f"{'{:.2f}'.format(item[6] * 100)} cm", (item[0][0] - 50, item[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, measurement_color, 2, cv2.LINE_AA)
                                img_l = cv2.putText(img_l, f"{item[7]}", (item[1][0] - 50, item[1][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, number_color, 2, cv2.LINE_AA)
                                img_l = cv2.putText(img_l, f"{'{:.2f}'.format(item[6] * 100)} cm", (item[1][0] - 50, item[1][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, measurement_color, 2, cv2.LINE_AA)

                                img_r = cv2.circle(img_r, (item[2][0], item[2][1]), radius=3, color=(255, 255, 0), thickness=-1)
                                img_r = cv2.circle(img_r, (item[4][0], item[4][1]), radius=3, color=(255, 255, 0), thickness=-1)
                                img_l = cv2.circle(img_l, (item[3][0], item[3][1]), radius=3, color=(255, 255, 0), thickness=-1)
                                img_l = cv2.circle(img_l, (item[5][0], item[5][1]), radius=3, color=(255, 255, 0), thickness=-1)

                                # Dibujar línea entre los puntos
                                img_r = cv2.line(img_r, (item[2][0], item[2][1]), (item[4][0], item[4][1]), (0, 255, 0), 2)
                                img_l = cv2.line(img_l, (item[3][0], item[3][1]), (item[5][0], item[5][1]), (0, 255, 0), 2)

                        
                        
                obj_counter += 1  # Incrementar contador de objetos
            #obtener promedio de las longitudes
            promedio = sum([item[6] for item in lengths]) / len(lengths)

            # Dibujar texto en la imagen
            add_text_box(img_r, f"Numero de objetos detectados: {obj_counter - 1}", (img_r.shape[1] - 200,img_r.shape[0]-200))
            pos = (img_r.shape[1] - 275, img_r.shape[0] - 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            line_type = cv2.LINE_AA
            text = f"Longitud promedio: {'{:.2f}'.format(promedio * 100)} cm"
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_origin = (pos[0] - text_size[0], pos[1])
            img_r = cv2.putText(img_r, text, (text_origin[0] + 5, text_origin[1] + text_size[1] + 5),font, font_scale, (0, 255, 255), font_thickness, line_type)

            add_text_box(img_l, f"Numero de objetos detectados: {obj_counter - 1}", (img_l.shape[1] - 200,img_l.shape[0]-200))
            img_l = cv2.putText(img_l, text, (text_origin[0] + 5, text_origin[1] + text_size[1] + 5),font, font_scale, (0, 255, 255), font_thickness, line_type)

            cv2.imwrite(f"{save_imgs_dir}/{r_img_name}", img_r)
            cv2.imwrite(f"{save_imgs_dir}/{l_img_name}", img_l)
            lengths = []

