import os
import numpy as np
import cv2
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'E:\Face_rec\MyData\Faces\subhs'

def list_img(path):

	img_path = [ os.path.join(path, f) for f in os.listdir(path) ]

	faces = []
	IDs = []

	for img in img_path:

		face=Image.open(img).convert('L')

		face = np.array(face, 'uint8')

		ID = int(os.path.split(img)[-1].split(".")[0])

		faces.append(face)
		IDs.append(ID)

	# print(faces)
	# print(IDs)
	return np.array(IDs), faces

Ids, faces1 = list_img(path)

# print(Ids)
# print(faces)

recognizer.train(faces1,Ids)

recognizer.write('trainr.yml')