import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    print("[INFO] Mediapipe detection")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    print("[INFO] model Processed: {}".format(results))
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(frame, results):
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius = 1), 
    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius = 1))

    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius = 1), 
    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius = 1))

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius = 4), 
    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius = 2))

    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius = 2), 
    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius = 2))

    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius = 4), 
    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius = 2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z ] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z ] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join("MP_DATA")
actions = np.array(["hello", "thanks", "iloveyou"])
no_seq = 30
seq_len = 30

for action in actions:
    for seq in range(no_seq):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
        except:
            pass

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_seq):
        window = []
        for frame_num in range(seq_len):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num) ))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

log_dir = os.path.join("Logs")
tb_callback =  TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences = True, activation="relu", input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences = True, activation="relu"))
model.add(LSTM(64, return_sequences = False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

history = model.fit(x_train, y_train, epochs=2, callbacks=[tb_callback])



def detection():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        for action in actions:
            for seq in range(no_seq):
                for frame_num in range(seq_len):
                    success, frame = cap.read()
                    if not success:
                        print("[INFO] Ignoring empty camera frame.")
                        continue
                    global results
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    # print(results)
                    if frame_num == 0:
                        cv2.putText(image, "STARTING COLLECTION", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, "Collecting frames for {} video number {}".format(action, seq), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow("OpenCV Feed", image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, "Collecting frames for {} video number {}".format(action, seq), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow("OpenCV Feed", image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.join.path(DATA_PATH, action, str(seq), str(frame_num ))
                    np.save(npy_path, keypoints)

                    # cv2.imshow("frame", image)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
        cap.release()
        cv2.destroyAllWindows() 

detection()

