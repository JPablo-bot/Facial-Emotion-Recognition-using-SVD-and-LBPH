import os
import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
from numpy import savetxt

def getPhotos(numPe, numPhotos):
    baseData = np.zeros((162*162,(numPe*numPhotos)))
    for person in range(numPe):
        for photo in range(numPhotos):
            root = DBPath+'/'+'s'+ str(person+1)+'/' +f'/rotro_{photo}.jpg'
            ima = color.rgb2gray(io.imread(root))
            signal = np.ravel(ima) # matrix to vector
            baseData[:,((numPe*photo)+person)] = signal
    return baseData


DBPath = 'Emociones/DBCompleta'#Cambia a la ruta donde hayas almacenado Data
listaSs = os.listdir(DBPath)
print('Lista de carpetas: ', listaSs)
    
DB = getPhotos(4,100)

DBmean = np.array([np.mean(DB,axis=1)])
savetxt('DBmean.txt', DBmean)
one = np.ones((1,DB.shape[1]),dtype='float32')
X = DB-(DBmean.T*one) #BDP=disp
cova = np.dot(X.T,X) #MC= np.dot()
[ev,ef] = np.linalg.eig(cova)
Xemph = np.dot(X,ef) # Matriz de características (enfatizador de características)
savetxt('Xemph.txt', Xemph)


