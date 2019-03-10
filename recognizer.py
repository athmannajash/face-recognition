import cv2, os
import numpy as np
from PIL import Image
import pickle
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/traingData.yml')
cascadePath = "classifier/haarface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataset'

def getprofile(id):
    conn=sqlite3.connect("face.db")
    sql="SELECT * FROM user WHERE ID="+str(id)
    cursor=conn.execute(sql)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        profile=getprofile(id)
        if(profile!=None):
            cv2.cv.PutText(cv2.fromarray(im), str(profile[1]),(x,y+h+20),font, 255)
            cv2.cv.PutText(cv2.fromarray(im), str(profile[2]),(x,y+h+50),font, 255)
            cv2.cv.PutText(cv2.fromarray(im), str(profile[3]),(x,y+h+80),font, 255)
            cv2.cv.PutText(cv2.fromarray(im), str(profile[4]),(x,y+h+100),font, 255)
    cv2.imshow('im',im)
    if cv2.waitKey(1) & 0xFF == ord('p'):#press p to terminate the window
        break


cam.release()
cv2.destroyAllWindows()
