import numpy as np
import cv2 as cv
import os

cam = 'Cam_sup_d'

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, 0.000001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((12*18,3), np.float32)*0.025
objp[:,:2] = np.mgrid[0:12,0:18].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = os.listdir(f'./Calibration/{cam}')

"""
for fname in images:
    if fname.endswith('.jpg'):
        print(f'./Calibration/{cam}/{fname}')
        img = cv.imread(f'./Calibration/{cam}/{fname}')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (18,12), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (18,12), corners2, ret)
            dim = (int(img.shape[1] * 50 / 100), int(img.shape[0] * 50 / 100))
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            cv.imshow('img', img)
            cv.waitKey(0)
            print(fname)
    cv.destroyAllWindows()
"""

for fname in images:
    if fname.endswith('.jpg'):
        print(f'./Calibration/{cam}/{fname}')
        img = cv.imread(f'./Calibration/{cam}/{fname}')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCornersSB(gray, (12, 18), None, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY | cv.CALIB_CB_LARGER | cv.CALIB_CB_MARKER)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (12,18), corners, ret)
            dim = (int(img.shape[1] * 50 / 100), int(img.shape[0] * 50 / 100))
            img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            cv.imshow('img', img)
            cv.waitKey(0)
            print(fname)
    cv.destroyAllWindows()

 #Get camera parameters for undistorting
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(F"RMS: {ret}")

#Save the parameters to CSV (To read back as a numpy array use np.loadtxt)
np.savetxt(f'Calibration/{cam}/mtx_{cam}.csv', mtx, delimiter=',')
np.savetxt(f'Calibration/{cam}/dist_{cam}.csv', dist, delimiter=',')

"""
#Refine camera parameters for the current image
img = cv.imread('./Calibration/Images/Lat 5-6-2023, 10-40-47.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow('calibrated image', dst)
cv.waitKey(500)
cv.imwrite('./Calibration/Chessboard/calibrate_result.jpg', dst)
"""