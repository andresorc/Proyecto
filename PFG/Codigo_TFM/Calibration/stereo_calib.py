import cv2 as cv
import os
import numpy as np

dir = 'Calibration/Stereo/Images'

mtx_cam_sup_d = np.loadtxt('Calibration/Cam_sup_d/mtx_Cam_sup_d.csv', delimiter=',')
dist_cam_sup_d = np.loadtxt('Calibration/Cam_sup_d/dist_Cam_sup_d.csv', delimiter=',')

mtx_cam_sup_i = np.loadtxt('Calibration/Cam_sup_i/mtx_Cam_sup_i.csv', delimiter=',')
dist_cam_sup_i = np.loadtxt('Calibration/Cam_sup_i/dist_Cam_sup_i.csv', delimiter=',')

#Se puede probar a cambiar estos parametros
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, 0.000001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*14,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:14].T.reshape(-1,2)
objp = objp*0.0245

# Arrays to store object points and image points from all the images.
imgpoints_cam_sup_d = [] # 2d points in image plane.
imgpoints_cam_sup_i = []

objpoints = [] # 3d point in real world space

images_names = os.listdir(dir)
images_names = sorted(images_names)
cam_sup_i_images_names = images_names[:len(images_names)//2]
cam_sup_d_images_names = images_names[len(images_names)//2:]
 
_im = cv.imread(f'{dir}/{cam_sup_d_images_names[0]}')
width = _im.shape[1]
height = _im.shape[0]

for im1, im2 in zip(cam_sup_i_images_names, cam_sup_d_images_names):
        
        print(f"Leyendo imágenes {im1}, {im2}")
        
        frame1 = cv.imread(f'{dir}/{im1}', 1)
        frame2 = cv.imread(f'{dir}/{im2}', 1)

        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        c_ret1, corners1 = cv.findChessboardCornersSB(gray1, (3, 3), None, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY | cv.CALIB_CB_LARGER | cv.CALIB_CB_MARKER)
        c_ret2, corners2 = cv.findChessboardCornersSB(gray2, (3, 3), None, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY | cv.CALIB_CB_LARGER | cv.CALIB_CB_MARKER)
 
        if c_ret1 == True and c_ret2 == True:
            
            #corners1_2 = cv.cornerSubPix(gray1, corners1, (5,5),(-1,-1), criteria)
            #corners2_2 = cv.cornerSubPix(gray2, corners2, (5,5),(-1,-1), criteria)
 
            cv.drawChessboardCorners(frame1, (9,14), corners1, c_ret1)
            dim = (int(frame1.shape[1] * 50 / 100), int(frame1.shape[0] * 50 / 100))
            frame1 = cv.resize(frame1, dim, interpolation = cv.INTER_AREA)
            cv.imshow('img', frame1)
            cv.waitKey(500)
 
            cv.drawChessboardCorners(frame2, (9,14), corners2, c_ret2)
            dim = (int(frame2.shape[1] * 50 / 100), int(frame2.shape[0] * 50 / 100))
            frame2 = cv.resize(frame2, dim, interpolation = cv.INTER_AREA)
            cv.imshow('img2', frame2)
            k = cv.waitKey(0)

            print(f"{im1},{im2}")
 
            objpoints.append(objp)
            imgpoints_cam_sup_d.append(corners2)
            imgpoints_cam_sup_i.append(corners1)

#stereocalibration_flags = cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL
stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_cam_sup_i, imgpoints_cam_sup_d, mtx_cam_sup_i, dist_cam_sup_i,
                                                                 mtx_cam_sup_d, dist_cam_sup_d, (width, height), criteria = criteria, flags = stereocalibration_flags)
print(R)
print(T)
#RMS
print(ret)

np.savetxt(f'Calibration/Stereo/R_stereo.csv', R, delimiter=',')
np.savetxt(f'Calibration/Stereo/T_stereo.csv', T, delimiter=',')