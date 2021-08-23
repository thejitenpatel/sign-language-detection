import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import mediapipe as mp
import os


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    print("[INFO] Mediapipe detection")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeble = False
    results = model.process(image)
    print("[INFO] model Processed: {}".format(results))
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def detection():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success:
                print("[INFO] Ignoring empty camera frame.")
                continue
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            cv2.imshow("frame", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows() 

detection()