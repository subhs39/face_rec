import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0
name=input("please enter a unique id (eg 1, 2, ..)") #this will serve as label for the classifier

def face_collect(face,count):
    face = cv2.resize(face, (200, 200))
    face = cv2.equalizeHist(face)
    file_name_path='./Faces/subhs/'+name+'.'+str(count)+'.jpg'
    cv2.imwrite(file_name_path, face)



while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) is not 0:
        count = count+1
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h, x:x+w]
        cv2.putText(img,str(count),(50,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (200,255,155), 5, cv2.LINE_AA)
        face_collect(face, count)
        
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF== ord('q') or count is 100:
        break

cap.release()
cv2.destroyAllWindows()