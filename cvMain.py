import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398] #left eye landmarks
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246] #right eye landmarks
RIGHT_IRIS = [474,475,476,477] #right iris landmarks
LEFT_IRIS = [469,470,471,472] #left iris landmarks
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]
def euclidean_distance(point1,point2):
    x1,y1 = point1.ravel()
    x2,y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def iris_position(iris_center,right_point,left_point):
    center_to_right_dist = euclidean_distance(iris_center,left_point)
    total_distance = euclidean_distance(right_point,left_point)
    ratio = center_to_right_dist/total_distance
    print(pyautogui.position().x)
    iris_position=""
    if ratio <= 0.40:
        iris_position="left"
        pyautogui.dragRel(0, -100, duration = 0.3)
        print(pyautogui.position().x)
    elif ratio>0.40 and ratio<=0.65:
        iris_position = "center"
    elif ratio>0.65:
        pyautogui.dragRel(0, 100, duration = 0.3)
        print(pyautogui.position().x)
        iris_position = "right"

    return iris_position , ratio


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)

    lvDistance = euclidean_distance(lv_top, lv_bottom)
    lhDistance = euclidean_distance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

cap = cv.VideoCapture(0)
time.sleep(1)
pyautogui.FAILSAFE = False
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret , frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame,1)
        rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x,p.y],[img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark ])
            #print(mesh_points.shape)
            #cv.polylines(frame,[mesh_points[LEFT_IRIS]],True,(0,255,0),1,cv.LINE_AA)
            #cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx , l_cy] ,dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame,center_left,int(l_radius),(255,0,255),1,cv.LINE_AA)
            cv.circle(frame,mesh_points[R_H_RIGHT][0],2,(255,0,255),1,cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 2, (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 2, (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT][0], 2, (0, 255, 0), 1, cv.LINE_AA)
            iris_pos , ratio = iris_position(
                center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0]
            )

            print(iris_pos)
        cv.imshow('img',frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break

cap.release()
cv.destroyAllWindows()
