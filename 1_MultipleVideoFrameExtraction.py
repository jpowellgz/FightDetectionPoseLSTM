import cv2
import os
import math
import time

#Call to the Open Pose Pose Estimation code by Vikas Gupta
from GuptaOpenPoseImplementation import *

#-----------------------------------------------------------------------
#Code modified from the OpenCV library
#Extracts frames from a set of videos and processes them with Open Pose
#-----------------------------------------------------------------------

#Time measuring
starttime= time.time()

#Definition of filters for edge detection

#filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
#filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
sobel1 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#filter2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#sobel2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

#Number of frames
extractedFrames = 10

#Total Violent videos
totalViolent = 100 #Depend on dataset
totalNonviolent = 100

for filenum in range(1,totalViolent):

# Filename = file location for each video, name is depending on dataset and id number
    filename="//[Insert file location here]//violent//v"+str(filenum)+".mp4"
    cam = cv2.VideoCapture(filename)
    print(filename)
    #Calculate total number of frames
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    
    #Calculate frames needed to extract 10 frames from the complete sequence
    frameTrue = math.floor(totalFrames/extractedFrames)
    # frame
    print(totalFrames)
    print(frameTrue)
    currentframe = 0
    counting =0

    while(True):
    	# reading from frame
        ret,frame = cam.read()
        num = currentframe+1
        div = num%frameTrue
        #If there is a frame to read
        if ret:
        	#If there have been less than 10 frames extracted
            if div==0 and counting<=extractedFrames:
    			# if video is still left continue creating images
    			print(str(counting)+' '+str(currentframe))
    			nam='./OpenPoseMatrices/v'+str(videonum)+'_'+str(framenum) #save matrix files to a folder
                name  = './OpenPoseImages/holder.jpg'
                counting = counting +1
                #print ('Creating...' + name)
                #scale_percent = 60 # percent of original size
                #width = int(img.shape[1] * scale_percent / 100)
                #height = int(img.shape[0] * scale_percent / 100)
                #dim = (width, height)

				#Preprocessing with filters defined previously
				#Horizontal sobel filters combination
				
                sob1 = cv2.filter2D(frame,-1,sobel1)
                sob2 = cv2.filter2D(frame,-1,sobel2)
                sumSobel = cv2.addWeighted(sob1,0.5,sob2,0.5,0)
                
                #Addition of original image with filtered image to improve quality for Open Pose
                fixed = cv2.addWeighted(frame,0.7,sumSobel,0.3,0
                
                # resize image
                resized = cv2.resize(fixed, (640,360), interpolation = cv2.INTER_AREA)
    			# writing the extracted images
                cv2.imwrite(name, resized)
                
                #Process the extracted frame with Open Pose
				openP(name,filenum,counting,nam)
    			# increasing counter so that it will
    			# show how many frames are created
            currentframe=currentframe+1
        else:
            print("End")
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    
for filenum in range(1,totalNonviolent):

# Filename = file location for each video, name is depending on dataset and id number
    filename="//[Insert file location here]//Dataset//nonviolent//nv"+str(filenum)+".mp4"
    cam = cv2.VideoCapture(filename)
    print(filename)
    #Calculate total number of frames
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    
    #Calculate frames needed to extract 10 frames from the complete sequence
    frameTrue = math.floor(totalFrames/extractedFrames)
    # frame
    print(totalFrames)
    print(frameTrue)
    currentframe = 0
    counting =0

    while(True):
    	# reading from frame
        ret,frame = cam.read()
        num = currentframe+1
        div = num%frameTrue
        #If there is a frame to read
        if ret:
        	#If there have been less than 10 frames extracted
            if div==0 and counting<=extractedFrames:
    			# if video is still left continue creating images
    			print(str(counting)+' '+str(currentframe))
    			nam='./OpenPoseMatrices/nv'+str(videonum)+'_'+str(framenum) #save matrix files to a folder
                name  = './OpenPoseImages/holder.jpg'
                counting = counting +1
                #print ('Creating...' + name)
                #scale_percent = 60 # percent of original size
                #width = int(img.shape[1] * scale_percent / 100)
                #height = int(img.shape[0] * scale_percent / 100)
                #dim = (width, height)

				#Preprocessing with filters defined previously
				#Horizontal sobel filters combination
				
                sob1 = cv2.filter2D(frame,-1,sobel1)
                sob2 = cv2.filter2D(frame,-1,sobel2)
                sumSobel = cv2.addWeighted(sob1,0.5,sob2,0.5,0)
                
                #Addition of original image with filtered image to improve quality for Open Pose
                fixed = cv2.addWeighted(frame,0.7,sumSobel,0.3,0
                
                # resize image
                resized = cv2.resize(fixed, (640,360), interpolation = cv2.INTER_AREA)
    			# writing the extracted images
                cv2.imwrite(name, resized)
                
                #Process the extracted frame with Open Pose
				openP(name,filenum,counting, nam)
    			# increasing counter so that it will
    			# show how many frames are created
            currentframe=currentframe+1
        else:
            print("End")
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()   
    
#Time measurement
exTime = (time.time()-starttime)
print("Execution time: "+str(exTime))
