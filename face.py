import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
cv2.namedWindow("athman", cv2.WINDOW_AUTOSIZE)
message = ""

while webcam.isOpened():
	
	_, frame = webcam.read()
	
	cv2.rectangle(frame, (100, 100), (530, 400), (150, 150, 0), 3)
	cv2.putText(frame, message, (95, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, (150, 150, 0), 2)

	cv2.imshow('athman', frame)
	key = cv2.waitkey(100) & 0xFF == 27
	if key not in [255, 27]:
		message += chr(key)
	elif key == 27:
		break

# release both video objects created
webcam.release()
cv2.destroyAllWindows()
