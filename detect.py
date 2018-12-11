import cv2
import os, sys
import numpy as np
import math
from matplotlib import pyplot as plt 

webcam = cv2.VideoCapture(0)
webcam.open(0)
folder = "people/" + raw_input('Person:').lower() #input name
cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
if not os.path.exists(folder):
	os.mkdir(folder)
	counter = 0
	timer = 0
	while counter < 10 : #take 10 pictures
		frame = webcam.get_frame()
		faces_coord = detector.detect(frame) #detect
		if len(faces_coord) and timer % 700 == 50: #every second or so
			faces = normalize_faces(frame, faces_coord) #norm pipeline
			cv2.imwrite(folder + '/' + str(counter) + '.jpg' , faces[0])
			plt_show(faces[0], "images saved:" + str(counter))
			clear_output(wait = True)# saved face in notebook
			counter += 1
		draw_rectangle(frame, faces_coord) # rectangle around the face
		cv2.imshow("Pydata tutorial", frame) #live feed in external
		cv2.waitkey(50)
		timer += 50
	cv2.destroyAllWindows()
else:
	print "This name already exists"
