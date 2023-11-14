import os
import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
from numpy import savetxt, loadtxt


DBmean = loadtxt('DBmean.txt')
Xemph = loadtxt('Xemph.txt')

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


DBPath = 'Emociones/DataBase'#Cambia a la ruta donde hayas almacenado Data
listaEmociones = os.listdir(DBPath)
print('Lista de emociones: ', listaEmociones)

labels = []
facesData = []
label = 0

for nameDir in listaEmociones:
    emotionsPath = DBPath + '/' + nameDir
    
    U,S,VT =  np.linalg.svd(Xemph,full_matrices=0)
    r = 50

    for fileName in os.listdir(emotionsPath):
          labels.append(label)
          gray = color.rgb2gray(plt.imread((emotionsPath+'/'+fileName),0))-np.reshape(DBmean,[162,162])
          Xgray = np.ravel(gray)

          imanew = DBmean+U[:,:r] @ U[:,:r].T @ Xgray
         
          facesData.append(imanew)
    label += 1
cont =0


savetxt('facesData.txt', facesData)

for k in range(0,400,200):  
    histTotal = np.zeros(((3*4*256),len(facesData)))
    for photo in range(k,k+int(len(facesData)/2),1):
        fotoLBPH = LBPH(np.reshape(facesData[photo],(162,162)))
        vecHist=exctHist(fotoLBPH)
        histTotal[:,photo]=np.reshape(vecHist,(3*4*256,))
    
    histMean = np.array([np.mean(histTotal,axis=1)])
    savetxt(f'histMean{cont}.txt', histMean)
    cont += 1



