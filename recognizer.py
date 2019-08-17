#!/usr/bin/env python3
import cv2, os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from PIL import Image, ImageTk

import pickle
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import _thread

#instanciating xml, yml
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/traingData.yml')
cascadePath = "classifier/haarface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataset'

#function to interact with sqlite3 database
def getprofile(id):
    conn = sqlite3.connect("face.db")
    sql = "SELECT * FROM user WHERE ID=" + str(id)
    cursor = conn.execute(sql)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


white = "#ffffff"
cream = "#fffdd0"
maxWidth = 1000
maxHeight = 600

# Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg=cream)
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth, maxHeight, 0, 0))
mainWindow.resizable(0, 0)
# mainWindow.overrideredirect(1)

mainFrame = Frame(mainWindow)
mainFrame.place(x=5, y=5)

# Capture video frames
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)

#instanciating webcam to variable cap
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX


#function to find face in each video frame
def show_frame():
    ret, im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        #cv2 drawing rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
        profile = getprofile(id)
        if profile is not None:
            #draw text around user face found
            cv2.putText(im, str(profile[1]), (x + 5 + w, y + 15), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[2]), (x + 5 + w, y + 45), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[3]), (x + 5 + w, y + 75), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[4]), (x + 5 + w, y + 105), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    #passing processed video to tkinter window
    img = Image.fromarray(im).resize((988, 500))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

#button to terminate main window
closeButton = Button(mainWindow, text = "CLOSE")
closeButton.configure(command= lambda: mainWindow.destroy())
closeButton.place(x=90,y=530)

#show_frame()  # Display
_thread.start_new_thread(show_frame, ())

mainWindow.mainloop()  # Starts GUI
