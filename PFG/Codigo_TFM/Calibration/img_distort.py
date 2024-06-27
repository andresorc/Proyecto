import numpy as np
import cv2 as cv
import os

def img_distort(dir, mtx_cam1, dist_cam1, mtx_cam2, dist_cam2, mtx_cam_lat, dist_cam_lat):

    mtx_cam_r = np.loadtxt(mtx_cam1, delimiter=',')
    dist_cam_r = np.loadtxt(dist_cam1, delimiter=',')

    mtx_cam_l = np.loadtxt(mtx_cam2, delimiter=',')
    dist_cam_l = np.loadtxt(dist_cam2, delimiter=',')

    mtx_cam_lat = np.loadtxt(mtx_cam_lat, delimiter=',')
    dist_cam_lat = np.loadtxt(dist_cam_lat, delimiter=',')

    for file in os.listdir(dir):

        if file.endswith('.jpg') and ('cam_sup_l' in file or 'cam_sup_r' in file or 'cam_lat' in file):

            img = cv.imread(f'{dir}{file}')
            h,  w = img.shape[:2]

            if 'cam_sup_l' in file:

                mtx = mtx_cam_r
                dist = dist_cam_r

            elif 'cam_sup_r':

                mtx = mtx_cam_l
                dist = dist_cam_l

            else: 
                mtx = mtx_cam_lat
                dist = dist_cam_lat

            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            cv.imwrite(f'{dir}Undistorted/{file.partition(".jpg")[0]}_undistorted.jpg', dst)



if __name__=='__main__':
    dir = 'Calibration/'
    img_distort(f'Depth_estimation/Images/', f'{dir}Cam_sup_d/mtx_cam_sup_d.csv', f'{dir}Cam_sup_d/dist_cam_sup_d.csv', f'{dir}Cam_sup_i/mtx_cam_sup_i.csv', f'{dir}Cam_sup_i/dist_cam_sup_i.csv', f'{dir}Cam_lat/mtx_cam_lat.csv', f'{dir}Cam_lat/dist_cam_lat.csv')