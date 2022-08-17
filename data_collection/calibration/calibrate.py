import numpy as np
import cv2
import os
import yaml

import sys
sys.path.insert(0, '.')
from config import *

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# 6 and 7 are the number of intersection points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


square_size = .02435 # squares are 2.435 cm^2
objp = objp * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


directory = config['path'] + "data_collection/calibration/selfie_calibration_data"
images = os.listdir(directory) 

for fname in images:
    img = cv2.imread(directory + "/" + fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # print(fname)
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (9,6), corners,ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

img = cv2.imread(config['path'] + 'data_collection/calibration/selfie_calibration_data/' + images[0])
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# save parameters for calibration
with open(config['path'] + "data_collection/calibration/selfie_calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow('calibresult',dst)
cv2.waitKey()

cv2.destroyAllWindows()

