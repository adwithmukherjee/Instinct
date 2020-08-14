import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import cv2
import dlib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cap = cv2.VideoCapture(0)

x, y = pkl.load(open('data/samples.pkl', 'rb'))

model = load_model('models/model.h5')

print(x.shape)
print(y.shape)

roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

print(roll.min(), roll.max(), roll.mean(), roll.std())
print(pitch.min(), pitch.max(), pitch.mean(), pitch.std())
print(yaw.min(), yaw.max(), yaw.mean(), yaw.std())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_val = std.transform(x_val)
x_test = std.transform(x_test)

def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1: return []
    
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points
        
def compute_features(face_points):
    #assert (len(face_points) == 68), "len(face_points) must be 68"
    n = len(face_points)
    face_points = np.array(face_points)
    features = []
    for i in range(n):
        for j in range(i+1, n):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)

#im = cv2.imread('data/lena.png', cv2.IMREAD_COLOR)
#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

while (True): 
    ret, im = cap.read(0)

    face_points = detect_face_points(im)

    #for x, y in face_points:
        #cv2.circle(im, (x, y), 1, (0, 255, 0), -1)
    
    features = compute_features(face_points)
    features = std.transform(features)
    y_pred = model.predict(features)
    roll_pred, pitch_pred, yaw_pred = y_pred[0]
    cv2.putText(im, ' Roll: {:.2f}°'.format(roll_pred), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )
    cv2.putText(im, 'Pitch: {:.2f}°'.format(pitch_pred), (50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )
    cv2.putText(im, '  Yaw: {:.2f}°'.format(yaw_pred), (50,150),cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1 )

    #cv2.imshow('pose detect', cv2.resize(im, (800,600)))
    print(' Roll: {:.2f}°'.format(roll_pred))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



roll_pred, pitch_pred, yaw_pred = y_pred[0]
#print(' Roll: {:.2f}°'.format(roll_pred))
#print('Pitch: {:.2f}°'.format(pitch_pred))
#print('  Yaw: {:.2f}°'.format(yaw_pred))
    
#plt.figure(figsize=(10, 10))
#plt.imshow(im)
#plt.show()