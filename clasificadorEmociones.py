import cv2
import os
import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
from numpy import savetxt, loadtxt

histMeanFelicidad = loadtxt('histMeanFelicidad.txt')
histMeanSorpresa = loadtxt('histMeanSorpresa.txt')

plt.figure()
plt.plot(np.linspace(0,3071,3072),histMeanFelicidad.T)
plt.figure()
plt.plot(np.linspace(0,3071,3072),histMeanSorpresa.T)

DBmean = loadtxt('DBmean.txt')
Xemph = loadtxt('Xemph.txt')

Emojis = [io.imread('1.png'), io.imread('2.png')]

# ----------------------------------- Capturar foto ---------------------------
if not os.path.exists('FacesTest'):
    print('Folder created: FacesTest')
    os.makedirs('FacesTest')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0

while True:
    a=0
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    k = cv2.waitKey(1)
    if k == 27:
        break
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(162,162), interpolation=cv2.INTER_CUBIC)
        if k == ord('s'):
            a=1
            cv2.imwrite('FacesTest/rotro_{}.jpg'.format(count),rostro)
            cv2.imshow('rotro_',rostro)
            count = count +1
    cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
    cv2.putText(frame,'Press s, to save the face',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if (a==1):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
def LBPH(image):
    newPhoto = np.zeros((162-2,162-2))
    kernel = np.zeros((3,3))
    for i in range(1,len(image)-1,1):
        for j in range(1,len(image)-1,1):
            kernel[0,0] = image[i-1,j-1]
            kernel[0,1] = image[i-1,j]
            kernel[0,2] = image[i-1,j+1]
            kernel[1,0] = image[i,j-1]
            kernel[1,1] = image[i,j]
            kernel[1,2] = image[i,j+1]
            kernel[2,0] = image[i+1,j-1]
            kernel[2,1] = image[i+1,j]
            kernel[2,2] = image[i+1,j+1]
            newKernel = np.where(kernel < image[i,j], 0, 1)
            binaryKernel = [str(newKernel[0,0])+str(newKernel[0,1])+str(newKernel[0,2])+str(newKernel[1,0])+
                               str(newKernel[1,2])+str(newKernel[2,0])+str(newKernel[2,1])+str(newKernel[2,2])]
            newPhoto[i-1,j-1] = int(binaryKernel[0], 2)
    return newPhoto


def exctHist (nPhoto):
    histogram = np.zeros((1,3*4*256))
    p=0
    for i in range(0,len(nPhoto),20):
        for j in range(0,len(nPhoto),20):
            vecHistogram = np.zeros((1,256))
            for k in range(i,i+20,1):
                for l in range(j,j+20,1):
                    vecHistogram[0,int(nPhoto[k,l])] = int(vecHistogram[0,int(nPhoto[k,l])]+1)
            if ((i==5*20 or i==6*20 or i==7*20) and (j==2*20 or j==3*20 or j==4*20 or j==5*20)):    
                histogram[0,p*256:(p*256+256)]=vecHistogram
                p+=1
    return histogram


DBPath = 'FacesTest' #Cambia a la ruta donde hayas almacenado la foto
listaFoto = os.listdir(DBPath)
print('Lista de fotos: ', listaFoto)


U,S,VT =  np.linalg.svd(Xemph,full_matrices=0)
r = 100

gray = color.rgb2gray(plt.imread((DBPath+'/'+listaFoto[0]),0))-np.reshape(DBmean,[162,162])
Xgray = np.ravel(gray)
imaNew = DBmean+U[:,:r] @ U[:,:r].T @ Xgray      

imaLBPH = LBPH(np.reshape(imaNew,(162,162)))
vecHistIma=exctHist(imaLBPH)

# --------------------------- Fase de prueba -------------------------------

plt.subplot(1,2,1)
plt.imshow(gray,cmap = 'gray')
plt.title('Buscando emoción...')

distancia = np.zeros(2)
distancia[0] = np.linalg.norm(histMeanFelicidad-vecHistIma) #distancia Euclidiana
distancia[1] = np.linalg.norm(histMeanSorpresa-vecHistIma) #distancia Euclidiana
    
emocion = np.argmin(distancia)
plt.subplot(1,2,2)
plt.imshow(Emojis[emocion],cmap='gray')
plt.show(block = False)
plt.title('Se encontró que la persona está...')

plt.figure()
plt.imshow(imaLBPH,cmap='gray')


plt.figure()
plt.imshow(np.reshape(imaNew,(162,162)),cmap='gray')


plt.figure()
plt.plot(np.linspace(0,3071,3072),vecHistIma.T)









