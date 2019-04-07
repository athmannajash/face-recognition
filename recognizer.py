import cv2, os
import numpy as np
from PIL import Image
import pickle
import sqlite3
import matplotlib.pyplot as plt

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
            cv2.putText(im, str(profile[1]),(x+5+w,y+15),font,0.8, (255,0,0),2,cv2.LINE_AA)
            cv2.putText(im, str(profile[2]),(x+5+w,y+45),font, 0.8,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(im, str(profile[3]),(x+5+w,y+75),font,0.8, (255,0,0),2,cv2.LINE_AA)
            cv2.putText(im, str(profile[4]),(x+5+w,y+105),font, 0.8, (255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Machine',im)
    if cv2.waitKey(1) & 0xFF == ord('p'):#press p to terminate the window
        break


cam.release()
cv2.destroyAllWindows()




#import cv2
i#mport matplotlib.pyplot as plt

#img = cv2.imread('img.jpg',0)

#plt.imshow(img, cmap='gray')
#plt.show()
