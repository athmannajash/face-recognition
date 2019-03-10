#this block identifies a face in a given picture by use of haarcascade defined in openCV
import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3

detector= cv2.CascadeClassifier('classifier/haarface.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,69,255),2)

    cv2.imshow('detect face',img)
    if cv2.waitKey(1) & 0xFF == ord('p'):#press p to terminate the window
        break

cap.release()
cv2.destroyAllWindows()
