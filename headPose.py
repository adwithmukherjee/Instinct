import cv2
import numpy as np
import dlib
import time
import math
from math import hypot
from pynput.mouse import Button, Controller
print("Loading")
cap = cv2.VideoCapture(0)
time.sleep(2.0);

calibrated = False 

mouse = Controller()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

landmark_indices = [33,8,45,36,54,48]

right_eye_indices = [42, 43, 44, 45, 46, 47]
left_eye_indices = [36, 37, 38, 39, 40, 41]

height = 1900
width = 3000

def midpoint(p1x ,p1y, p2x, p2y):
    return int((p1x + p2x)/2), int((p1y + p2y)/2)

def get_blinking_ratio(eye_points):
    left_point = (eye_points[0][0], eye_points[0][1])
    right_point = (eye_points[3][0], eye_points[3][1])
    center_top = midpoint(eye_points[1][0], eye_points[1][1], eye_points[2][0], eye_points[2][1])
    center_bottom = midpoint(eye_points[5][0], eye_points[5][1], eye_points[4][0], eye_points[4][1])

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def detect_face_points(image):
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []
    
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    right_eye_points = []
    left_eye_points = []

    for i in range(len(landmark_indices)):
        j = landmark_indices[i]
        x, y = dlib_points.part(j).x, dlib_points.part(j).y
        face_points.append(np.array([x, y]))

    for i in range(len(right_eye_indices)):
        j = right_eye_indices[i]
        x, y = dlib_points.part(j).x, dlib_points.part(j).y
        right_eye_points.append(np.array([x, y]))

    for i in range(len(left_eye_indices)):
        j = left_eye_indices[i]
        x, y = dlib_points.part(j).x, dlib_points.part(j).y
        left_eye_points.append(np.array([x, y]))

    return face_points, right_eye_points, left_eye_points

# Read Image

model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])
x_center = width/2
y_center = height/2

prev_mousex = x_center
prev_mousey = y_center

threshold = 50

while(True): 

    


    _,im = cap.read()
    size = im.shape

    im = cv2.resize(im, (1200, 675))
    


    points, right_eye_points, left_eye_points = detect_face_points(im)

   

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (points[0][0], points[0][1]),     # Nose tip
                                (points[1][0], points[1][1]),     # Chin
                                (points[2][0], points[2][1]),     # Left eye left corner
                                (points[3][0], points[3][1]),     # Right eye right corne
                                (points[4][0], points[4][1]),     # Left Mouth corner
                                (points[5][0], points[5][1])     # Right mouth corner
                            ], dtype="double")

    # 3D model points.
    



    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    #print("Camera Matrix :\n {0}".format(camera_matrix));

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    #print("Rotation Vector:\n {0}".format(rotation_vector))
    #print("Translation Vector:\n {0}".format(translation_vector))
    #print(translation_vector.shape)


    # print("x")
    # print(translation_vector[0,0])
    # print("--------")
    # print(translation_vector[1,0])


    #cv2.putText(im, ' Translation X: {:.2f}°'.format(translation_vector[0,0]), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )
    #cv2.putText(im, ' Translation Y: {:.2f}°'.format(translation_vector[1,0]), (50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )
    #cv2.putText(im, ' Translation Z: {:.2f}°'.format(translation_vector[0][2]), (50,150),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )

    #cv2.putText(im, ' Rotation X: {:.2f}°'.format(rotation_vector[0][0]), (50,200),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    # if not calibrated: 
    #     print("Calibrating. Get Ready...")
    #     time.sleep(2.0)
    #     print("Top Left")
    #     time.sleep(3.0)
    #     left = translation_vector[0,0]
    #     top = translation_vector[1,0]
    #     print("Bottom Right")
    #     time.sleep(3.0)
    #     right = translation_vector[0,0]
    #     bottom = translation_vector[1,0]


    #     print(left)
    #     print(right)
    #     print(top)
    #     print(bottom)
        
    #     print("Calibrated. ")
    #     calibrated = True


    right = 140
    left = -80
    top = -300
    bottom = -430

    
    


    

    mousex = (translation_vector[0,0])*width/(right-left) + width/2 - right*width/(right-left)
    mousey = (translation_vector[1,0])*height/(top-bottom) + height/2 - top*height/(top-bottom)

    mousex = int(x_center + mousex)
    mousey = int(y_center - mousey)

    if (math.sqrt((mousex - prev_mousex)**2 + (mousey - prev_mousey)**2) < 100 ):
        
        mousex = prev_mousex
        mousey = prev_mousey

    prev_mousex = mousex
    prev_mousey = mousey
    

    

    #cv2.circle(im, (mousex, mousey), 20, (255,0,0), -1)

    mouse.position = (mousex,mousey)

    
    is_blinking_right = get_blinking_ratio(right_eye_points)
    is_blinking_left = get_blinking_ratio(left_eye_points)
    is_blinking = (is_blinking_left+is_blinking_right)/2

    print(is_blinking)

    if(is_blinking > 5.7): 
        #mouse.press(Button.left)
        #mouse.release(Button.left)
        print("BLINK")
    
    

    #(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    #for p in image_points:
        #cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    #p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    #cv2.line(im, p1, p2, (255,0,0), 2)

    


    # Display image
    #cv2.imshow("Output", cv2.resize(im, (1800,1000)));

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows(); 