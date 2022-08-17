import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import math
import time 

import sys
sys.path.insert(0, '.')
from config import *

NUMTAGS = 2
TAGSIZE_GLASSES = 0.0301625
TAGSIZE_WALL = 0.12779375

BASE_VECTOR = [0,0,1]

device = 1
cap = cv2.VideoCapture(device)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

camera_dim = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def get_calibration():
    with open(config['path'] + "data_collection/calibration/selfie_calibration_matrix.yaml", "r") as stream:
        try:
            calibration_data = yaml.safe_load(stream)
            mtx = np.asarray(calibration_data['camera_matrix'])
            dist = np.asarray(calibration_data['dist_coeff'])
            return mtx, dist
        except yaml.YAMLError as exc:
            print(exc)
            exit()


def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def angleWrap180(angle):
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 180
    return angle


def angleWrapPi(angle):
    while angle > math.pi:
        angle -= math.pi
    while angle <= -math.pi:
        angle += math.pi
    return angle

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def rpy_decomposition(rvec):
    rotM = np.zeros(shape=(3,3))
    cv2.Rodrigues(rvec, rotM, jacobian = 0)
    return cv2.RQDecomp3x3(rotM)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate_vectors(v, angle):
    xr, zu, yf = v[0], v[1], v[2]
    
    angle = angle * (math.pi / 180) * -1

    angle += math.atan2(zu, xr)

    radius = math.sqrt(xr ** 2 + zu ** 2)

    xr = radius * math.cos(angle)
    zu = radius * math.sin(angle)

    return [xr, zu, yf]

def getGlassesAngle(corners, ids):
    if np.all(ids is not None):  # If there are markers found by detector
        glasses_corners = []
        for i in range(len(ids)):
            if ids[i] == 1:
                glasses_corners.append(corners[i])
        
        aruco.drawDetectedMarkers(frame, glasses_corners)  # Draw A square around the markers

        if len(glasses_corners) == 2:
            rotation = []
            vectors = []
            for i in range(0, len(glasses_corners)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(glasses_corners[i], TAGSIZE_GLASSES, CAMERAMATRIX,
                                                                        DISTORTIONCOEFFICIENTS)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawAxis(frame, CAMERAMATRIX, DISTORTIONCOEFFICIENTS, rvec, tvec, 0.01)  # Draw Axis
                # Convert rvec to rotation matrix
                rotMat, jacob = cv2.Rodrigues(rvec)
                # Convert rotation matrix to Euler angles
                rotEuler = rotationMatrixToEulerAngles(rotMat)
                rotEuler = [i * (180.0 / float(format(math.pi, '.5f'))) for i in rotEuler]

                rotation.append(rotEuler[2])
                vectors.append(np.matmul(np.array(rotMat), np.array(BASE_VECTOR)))

            angle = angle_between(vectors[0], vectors[1]) * (180 / math.pi)

            if angle < 3:
                return (-1 * np.mean(vectors, axis=0), np.mean(rotation))
    return (None, None)


def getWallAngle(corners, ids):
    if np.all(ids is not None):  # If there are markers found by detector
        glasses_corners = []
        for i in range(len(ids)):
            if ids[i] == 0:
                glasses_corners.append(corners[i])
        
        aruco.drawDetectedMarkers(frame, glasses_corners)  # Draw A square around the markers

        if len(glasses_corners) == 2:
            rotation = []
            vectors = []
            for i in range(0, len(glasses_corners)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(glasses_corners[i], TAGSIZE_WALL, CAMERAMATRIX,
                                                                        DISTORTIONCOEFFICIENTS)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawAxis(frame, CAMERAMATRIX, DISTORTIONCOEFFICIENTS, rvec, tvec, 0.05)  # Draw Axis
                # Convert rvec to rotation matrix
                rotMat, jacob = cv2.Rodrigues(rvec.flatten())
                # Convert rotation matrix to Euler angles
                rotEuler = rotationMatrixToEulerAngles(rotMat)
                rotEuler = [i * (180.0 / float(format(math.pi, '.5f'))) for i in rotEuler]

                rotation.append(rotEuler[2])
                vectors.append(np.matmul(np.array(rotMat), np.array(BASE_VECTOR)))

            angle = angle_between(vectors[0], vectors[1]) * (180 / math.pi)

            if angle < 3:
                return (-1 * np.mean(vectors, axis=0), np.mean(rotation))
    return (None, None)

CAMERAMATRIX, DISTORTIONCOEFFICIENTS = get_calibration()
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,camera_dim,1,camera_dim)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

f = open(config['path'] + "data_collection/orientation_data.txt", "a")

while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=CAMERAMATRIX, distCoeff=DISTORTIONCOEFFICIENTS)

    glasses, g_roll = getGlassesAngle(corners, ids)
    wall, w_roll = getWallAngle(corners, ids)

    if wall is not None and glasses is not None:
        glasses = unit_vector(glasses)
        wall = unit_vector(wall)

        yaw_rad = math.atan2(wall[0], wall[2]) - math.atan2(glasses[0], glasses[2])
        pitch_rad = math.atan2(wall[1], wall[2]) - math.atan2(glasses[1], glasses[2])

        yaw_deg = yaw_rad * (180 / math.pi)
        pitch_deg = pitch_rad * (180 / math.pi)

        yaw = angleWrap180(yaw_deg)
        pitch = angleWrap180(pitch_deg)
        roll = angleWrap180(g_roll - w_roll)

        z = math.cos(yaw * (math.pi / 180)) * math.cos(pitch * (math.pi / 180))
        x = -math.sin(yaw * (math.pi / 180)) * math.cos(pitch * (math.pi / 180))
        y = -math.sin(pitch * (math.pi / 180))

        vectors = rotate_vectors([x,y,z], roll)

        pitch_rad = math.atan2(vectors[0], vectors[2])
        yaw_rad = math.atan2(vectors[1], vectors[2])

        pitch_deg = pitch_rad * (180 / math.pi)
        yaw_deg = yaw_rad * (-180 / math.pi)

        # print(yaw_deg, pitch_deg)
        f.write(str(int(time.time() * 1000)) + "_" + str(format(yaw_deg,".2f")) + "_" + str(format(pitch_deg,".2f"))  + "\n")

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)