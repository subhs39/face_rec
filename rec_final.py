import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)				#initializing webcam

detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('trainr.yml')						#loading the trained classifier
thresh = 90						#confidence

while True:

	_, img = cap.read()
	img = cv2.flip(img, 1)								#getting mirror image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		#drawing rectangles over the face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		#slicing and passing just the face to the trained model
		temp = gray[y:y+h,x:x+w]
		temp = cv2.equalizeHist(temp)		# normalizing the face
		Id,conf = detector.predict(temp)	# getting the label and the confidence

		# print(Id,conf)					# for debugging purposes i.e lets u see the
											# label and conf rate of the person identified
		if conf < thresh :	

			color = (0,255,0)		#Green
			if Id == 1:
				Id="Subham"
			if Id == 2:
				Id="Sudipta"
		else:

			color=(0,0,255)			#Red
			Id="Unknown"

		# Writing names of the identified person on the image
		cv2.putText(img,str(Id),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
	
	cv2.imshow('img',img)		
	
	if cv2.waitKey(1) & 0xFF== ord('q'):	#setting the frame rate and exit criteria
	    break			

cap.release()				#releasing the webcam
cv2.destroyAllWindows()		