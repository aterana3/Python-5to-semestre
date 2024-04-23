import cv2
import os
import numpy as np

dataPath = 'E:/VSCode/Reconocimiento facial/data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

#obtener los directorios de las personas
for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	#obtener las imágenes de las personas
	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		#asignar un label a cada imagen
		labels.append(label)
		#almacenar las imágenes en un arreglo
		facesData.append(cv2.imread(personPath+'/'+fileName, 0))
	label = label + 1


# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create() 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")

# Se le pasa el arreglo de imágenes y los labels
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
# Sirve para guardar el modelo entrenado
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')

print("Modelo almacenado...")