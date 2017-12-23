import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('trainr.yml')
thresh = 70

while True:

	_, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		temp = gray[y:y+h,x:x+w]
		temp = cv2.equalizeHist(temp)
		Id,conf = detector.predict(temp)
		# print(Id,conf)
		if conf < thresh :
			color = (0,255,0)
			if Id == 1:
				Id="subham"
			if Id == 2:
				Id="sachin"
		else:
			color=(0,0,255)
			Id="Unknown"
		cv2.putText(img,str(Id),(50,400), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 5, cv2.LINE_AA)
	cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF== ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()