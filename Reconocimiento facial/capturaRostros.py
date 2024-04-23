import cv2
import os
import imutils

personalName = "Anthony"
dataPath = 'E:/VSCode/Reconocimiento facial/data'
personalPath = dataPath + '/' + personalName

if not os.path.exists(personalPath):
    os.makedirs(personalPath)
    print('Carpeta creada: ', personalPath)

# Captura de video
# 0 - Indica el indice de camaras conectadas | 0 - para seleccionar la camara principal del dispositivo
# cv2.CAP_DSHOW - es un parametro opcional el cual especifica al backend la api de captura de video que se va a utilizar
cap = cv2.VideoCapture(0)

# Modelo pre-entrenado de reconocimiento de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    # Lectura de cada frame
    ret, frame = cap.read()
    if ret == False:
        break
    # Redimensionar el frame
    frame = imutils.resize(frame, width=640)
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # copia del frame
    auxFrame = frame.copy()

    # Detectar rostros
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        # Dibujar un rectangulo en la imagen
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        # Redimensionar el rostro
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        # Almacenar el rostro en la carpeta creada
        cv2.imwrite(personalPath + '/rostro_{}.jpg'.format(count),rostro)
        count = count + 1

    # Mostrar el frame
    cv2.imshow('frame',frame)

    # Esperar una tecla para salir
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

# Liberar la camara
cap.release()

# Cerrar todas las instancias de la ventana
cv2.destroyAllWindows()