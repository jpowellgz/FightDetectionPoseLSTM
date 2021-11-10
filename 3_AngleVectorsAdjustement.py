import numpy as np

#----------------------------------------------------
#Min max scaling of individual angle vectors
#Adjustment of angle vectors into a data matrix and a label matrix
#---------------------------------------------------


#Total number of videos, half violent and half nonviolent
size = 100
halfSize = int(size/2)

vMat = np.load('AngleMatrices/vMat.npy')
nvMat = np.load('AngleMatrices/nvMat.npy')

#Data matrix definition
dataMat = np.zeros((size,10,260))

#LAbel matrix definition
labelMat = np.zeros((size,))

for i in range(halfSize):
	a = 2*i
	b = (2*i)+1
	dataMat[a]=nvMat[i]
	dataMat[b]=vMat[i]
	labelMat[b]=1.0
print(dataMat.shape)


#Min max scaling
for k in range(size):
	for l in range(10):
		for m in range(13):
			ind1 = m*20
			ind2 = (m+1)*20
			maxim = 0
			minim = 10000
			for n in range(ind1,ind2):
				if dataMat[k][l][n]>maxim:
					maxim = dataMat[k][l][n]
				if dataMat[k][l][n]<minim:
					minim = dataMat[k][l][n]
			for n in range(ind1,ind2):
				dif = maxim - minim
				if dif>0:
					dataMat[k][l][n]=(dataMat[k][l][n]-minim)/dif
				else:
					dataMat[k][l][n]=0

np.save('TestMatrices/dataMat.npy',dataMat)
np.save('TestMatrices/labelMat.npy',labelMat)
