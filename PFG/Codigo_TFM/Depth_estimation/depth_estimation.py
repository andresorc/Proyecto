import json
import csv
import math
import cv2
import numpy as np

json_pts_dir = "Segmentation/head_tail_pts.json"
json_params_dir = 'Calibration/Stereo/params.json'
depth_csv_dir = 'Depth_estimation/depth.csv'
imgs_dir = "Depth_estimation/Images/Undistorted/Corrected/"
save_imgs_dir = "Results/"
lengths = []
draw_length = 1

json_pts_file = open(json_pts_dir)
pts_dict = json.load(json_pts_file)

json_params_file = open(json_params_dir)
params_dict = json.load(json_params_file)

depth_csv = open(depth_csv_dir, 'w')
writer = csv.writer(depth_csv, delimiter=';')

focal_length = params_dict['focal_length']
baseline = params_dict['baseline']
opt_cnt_x = params_dict['opt_cnt_x']
#opt_cnt_x = 2560/2
opt_cnt_y = params_dict['opt_cnt_y']

writer.writerow(["imagen"] + ["color"] + ["id_pez"] + ["profundidad de cabeza"] + ["profundidad cuerpo"] + ["profundidad cola"] + ["longitud proyectada estimada (px)"] + ["longitud proyectada estimada (m)"] + ["longitud real estimada (m)"] + ["longitud real (m)"] + ["error absoluto (mm)"])

for l_img_name in pts_dict.keys():
    if '_l_' in l_img_name:
        r_img_name = l_img_name.replace('_l_', '_r_')

        for id in pts_dict[l_img_name].keys():

            pts_l = pts_dict[l_img_name][id]
            centroid_l = pts_l[0]
            head_l = pts_l[1][0]
            tail_l = pts_l[1][1]
            color = pts_l[2]

            pts_r = pts_dict[r_img_name][id]
            centroid_r = pts_r[0]
            head_r = pts_r[1][0]
            tail_r = pts_r[1][1]

            if abs(centroid_l[1]-centroid_r[1]) > 10 or abs(head_l[1]-head_r[1]) > 10 or abs(tail_l[1]-tail_r[1]) > 10:
                error = 'Puntos mal seleccionados'
                writer.writerow([l_img_name.partition("_l_")[2]] + [color] + [id] + [error] + [""] + [""] + [""])
                print(f'*********{l_img_name.partition("_l_")[2]}/{id} -> cabeza {head_l[1]-head_r[1]} / cuerpo {centroid_l[1]-centroid_r[1]} / cola {tail_l[1]-tail_r[1]}')
                lengths.append([centroid_r, centroid_l, head_r, head_l, tail_r, tail_l,"NA"])
            else:
                real_length = 0

                if color == "rojo":
                    real_length = 0.035
                elif color == "blanco":
                    real_length = 0.04
                else:
                    real_length = 0.045

                head_depth = baseline*focal_length/abs(head_l[0]-head_r[0])
                body_depth = baseline*focal_length/abs(centroid_l[0]-centroid_r[0])
                tail_depth = baseline*focal_length/abs(tail_l[0]-tail_r[0])
                proyected_length = dist = np.linalg.norm(np.asarray(head_l)-np.asarray(tail_l))
                proyected_length_m = body_depth*proyected_length/focal_length
                real_estimated_length_m = math.sqrt(abs(head_depth-tail_depth)**2 + proyected_length_m**2)
                writer.writerow([l_img_name.partition("_l_")[2]] + [color] + [id] + [head_depth] + [body_depth] + [tail_depth] + [proyected_length] + [proyected_length_m] + [real_estimated_length_m] + [real_length] + [abs(real_estimated_length_m-real_length)*1000])
                print(f'{l_img_name.partition("_l_")[2]}/{id} -> cabeza {head_l[1]-head_r[1]} / cuerpo {centroid_l[1]-centroid_r[1]} / cola {tail_l[1]-tail_r[1]}')
                lengths.append([centroid_r, centroid_l, head_r, head_l, tail_r, tail_l, real_estimated_length_m])

            if draw_length:
                img_r = cv2.imread(f"{imgs_dir}{r_img_name}")
                img_l = cv2.imread(f"{imgs_dir}{l_img_name}")
                
                for item in lengths:

                    if item[6] == "NA":
                        img_r = cv2.putText(img_r, item[6], (item[0][0]-10, item[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                        img_l = cv2.putText(img_l, item[6], (item[1][0]-10, item[1][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    else:
                        img_r = cv2.putText(img_r, f"{'{:.2f}'.format(item[6]*100)} cm", (item[0][0]-50, item[0][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                        img_l = cv2.putText(img_l, f"{'{:.2f}'.format(item[6]*100)} cm", (item[1][0]-50, item[1][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

                    img_r = cv2.circle(img_r, (item[2][0], item[2][1]), radius=5, color=(0, 255, 255), thickness=-1)
                    img_r = cv2.circle(img_r, (item[4][0], item[4][1]), radius=5, color=(0, 255, 255), thickness=-1)
                    img_l = cv2.circle(img_l, (item[3][0], item[3][1]), radius=5, color=(0, 255, 255), thickness=-1)
                    img_l = cv2.circle(img_l, (item[5][0], item[5][1]), radius=5, color=(0, 255, 255), thickness=-1)

                cv2.imwrite(f"{save_imgs_dir}{r_img_name}", img_r)
                cv2.imwrite(f"{save_imgs_dir}{l_img_name}", img_l)
        
        lengths = []