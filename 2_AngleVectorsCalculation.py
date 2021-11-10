import numpy as np
import math
import time

#----------------------------------------------
#Code to extract angle vectors from Open Pose matrices
#Calculates the closest angle from 20 predetermined angles, equally spaced
#Stores all the vectors in two NumPy matrices, One for violent videos, other for nonviolent videos
#----------------------------------------------


#Start time measurement
startTime=time.time()

#Table for predetermined angle calculations based on slope
#Angles (degrees) = 0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342
#quadrant indexes I: 0-4,II:5-9, III:10-14,IV:15-19
tableX = [1.00,0.95,0.81,0.59,0.31,0.00,-0.31,-0.59,-0.81,-0.95,-1.00,-0.95,-0.81,-0.59,-0.31,0.00,0.31,0.59,0.81,0.95]
tableY = [0.00,0.31,0.59,0.81,0.95,1.00,0.95,0.81,0.59,0.31,0.00,-0.31,-0.59,-0.81,-0.95,-1.00,-0.95,-0.81,-0.59,-0.31]

#Image size
imgX=640
imgY=360

#Number of frames
numFrames = 10
#nam = 'testUT/opMatrices/op_golpe'+str(videonum)+'_'+str(framenum)+'.npy'
pairs = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],[1,8], [8,9], [9,10], [1,11], [11,12], [12,13],[1,0]]
    # POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   # ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   # ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   # ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"]
                   #13 pairs for nose to ankles

#VIOLENT VIDEOS MATRIX
#Produces a matrix of size [Total violent videos, 10, 260]

#Total of videos in the violent category
vTotal = 100 #Can be modified depending on Dataset


vMat = []
for videonum in range(1,vTotal+1):
    vidMat = []
    for framenum in range(1,numFrames+1):
        #nam = 'testVDDWF/matricesOpenPose/nv'+str(videonum)+'_'+str(framenum)+'.npy'
        #nam = 'testVDDWF/matricesOpenPose/v'+str(videonum)+'_'+str(framenum)+'.npy'
        nam = 'testPDNF/matricesOpenPose/v'+str(videonum)+'_'+str(framenum)+'.npy'
        print(nam)
        matrix = np.load(nam)
        #matrix with n people and 18 parts
        #20 angles per pair of body parts, 13 parts = 260 size
        anglesVector = np.zeros((13,20))
        #print(anglesVector.shape)
        nPeople = len(matrix)
        print("personas")
        print(nPeople)
        for pers in range(nPeople):
            #for every pair calculate difference in x, difference in y. approximate angle based on tables
            for pair in range(len(pairs)):
                ind1 = pairs[pair][0]
                ind2 = pairs[pair][1]
                x1 = matrix[pers][ind1][0]
                x2 = matrix[pers][ind2][0]
                y1 = matrix[pers][ind1][1]
                y2 = matrix[pers][ind2][1]
                #check that body part is not missing
                if (x1>0 or y1>0)and(x2>0 or y2>0):
                    deltaX = (x1-x2)
                    deltaY = (y1-y2)
                    hypotenuse = math.sqrt(deltaX*deltaX + deltaY*deltaY)
                    deltaX = deltaX/hypotenuse
                    deltaY = deltaY/hypotenuse
                    #lookup angle based on delta

                    if deltaX>0 and deltaY>0:
                        indX=0
                        indY=0
                        while(deltaX<tableX[indX] and indX<5):
                            indX = indX+1
                        while(deltaY>tableY[indY] and indY<5):
                            indY = indY+1
                    if deltaX<=0 and deltaY>0:
                        indX = 5
                        indY = 5
                        while(deltaX<tableX[indX] and indX<10):
                            indX = indX+1
                        while(deltaY<tableY[indY] and indY<10):
                            indY = indY+1
                    if deltaX<=0 and deltaY<=0:
                        indX = 10
                        indY = 10
                        while(deltaX>tableX[indX] and indX<15):
                            indX = indX+1
                        while(deltaY<tableY[indY]and indY<15):
                            indY = indY+1
                    if deltaX>0 and deltaY<=0:
                        indX = 15
                        indY = 15
                        while(deltaX>tableX[indX] and indX<19):
                            indX = indX+1
                        while(deltaY>tableY[indY] and indX<19):
                            indY=indY+1
                    #print("person "+str(pers))
                    #print("pair__________"+str(pair))
                    #print("angle is index")
                    #print(str(indX)+","+str(indY))
                    #add angle to vector
                    anglesVector[pair][indY]=anglesVector[pair][indY]+1
        #print("0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342")
        #print(anglesVector)
        ang = np.reshape(anglesVector, (260))
        vidMat.append(ang)
    vMat.append(vidMat)

#NONVIOLENT MATRIX
#Produces a matrix of size [Total nonviolent videos, 10, 260]

#Total of videos in the nonviolent category
nvTotal = 100 #Can be modified depending on Dataset

nvMat = []
for videonum in range(1,nvTotal+1):
    vidMat = []
    for framenum in range(1,numFrames+1):
        nam = 'OpenPoseMatrices/nv'+str(videonum)+'_'+str(framenum)+'.npy'
        print(nam)
        
        #matrix with n people and 18 parts
        matrix = np.load(nam)
        
        #20 angles per pair of body parts, 13 parts = 260 size
        anglesVector = np.zeros((13,20))
        #print(anglesVector.shape)
        nPeople = len(matrix)
        print("Number of People:")
        print(nPeople)
        for pers in range(nPeople):
            #for every pair calculate difference in x, difference in y. approximate angle based on tables
            for pair in range(len(pairs)):
                ind1 = pairs[pair][0]
                ind2 = pairs[pair][1]
                x1 = matrix[pers][ind1][0]
                x2 = matrix[pers][ind2][0]
                y1 = matrix[pers][ind1][1]
                y2 = matrix[pers][ind2][1]
                #check that body part is not missing
                if (x1>0 or y1>0)and(x2>0 or y2>0):
                    deltaX = (x1-x2)
                    deltaY = (y1-y2)
                    hypotenuse = math.sqrt(deltaX*deltaX + deltaY*deltaY)
                    deltaX = deltaX/hypotenuse
                    deltaY = deltaY/hypotenuse
                    #lookup angle based on delta

                    if deltaX>0 and deltaY>0:
                        indX=0
                        indY=0
                        while(deltaX<tableX[indX] and indX<5):
                            indX = indX+1
                        while(deltaY>tableY[indY] and indY<5):
                            indY = indY+1
                    if deltaX<=0 and deltaY>0:
                        indX = 5
                        indY = 5
                        while(deltaX<tableX[indX] and indX<10):
                            indX = indX+1
                        while(deltaY<tableY[indY] and indY<10):
                            indY = indY+1
                    if deltaX<=0 and deltaY<=0:
                        indX = 10
                        indY = 10
                        while(deltaX>tableX[indX] and indX<15):
                            indX = indX+1
                        while(deltaY<tableY[indY]and indY<15):
                            indY = indY+1
                    if deltaX>0 and deltaY<=0:
                        indX = 15
                        indY = 15
                        while(deltaX>tableX[indX] and indX<19):
                            indX = indX+1
                        while(deltaY>tableY[indY] and indX<19):
                            indY=indY+1
                    #print("person "+str(pers))
                    #print("pair__________"+str(pair))
                    #print("angle is index")
                    #print(str(indX)+","+str(indY))
                    #add angle to vector
                    anglesVector[pair][indY]=anglesVector[pair][indY]+1
        #print("0,18,36,54,72,90,108,126,144,162,180,198,216,234,252,270,288,306,324,342")
        #print(anglesVector)
        ang = np.reshape(anglesVector, (260))
        vidMat.append(ang)
    nvMat.append(vidMat)
    #vMat.append(vidMat)
print(str(len(nvMat))+','+str(len(nvMat[0]))+','+str(len(nvMat[0][0])))
print(str(len(vMat))+','+str(len(vMat[0]))+','+str(len(vMat[0][0])))

#Save the matrices as NumPy files for further processing
np.save('AngleMatrices/nvMat.npy',nvMat)
np.save('AngleMatrices/vMat.npy',vMat)

#Execution time measurement
exTime = (time.time()-startTime)
print("time = "+str(exTime))
