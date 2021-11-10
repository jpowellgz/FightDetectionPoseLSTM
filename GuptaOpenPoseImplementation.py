import cv2 as cv
import numpy as np
import argparse

#-----------------------------------------------------------------------
#Code provided by Vikas Gupta
#Refer to source previous to any modification to file: 
#Gupta, Vikas. Multi-Person Pose Estimation in OpenCV using OpenPose. https://www.learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose

#Location for the matrices and images files can be changed at the end with the nam and name variables





def openP(fil,videonum,framenum, nam):
    parser = argparse.ArgumentParser(
            description='This script is used to demonstrate OpenPose human pose estimation network '
                        'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                        'The sample and model are simplified and could be used for a single person on the frame.')
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--proto', help='Path to .prototxt')
    parser.add_argument('--model', help='Path to .caffemodel')
    parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                          'It could be (COCO, MPI, HAND) depends on dataset.')
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--scale', default=0.003922, type=float, help='Scale for blob.')
    args = parser.parse_args()
    #args.input='.\openpose\examples\media\COCO_val2014_000000000294.jpg'


	#Model selection: COCO or MPI
	#Width and height selection
    # args.proto='.\openpose\models\pose\mpi\pose_deploy_linevec.prototxt'
    # args.model='.\openpose\models\pose\mpi\pose_iter_160000.caffemodel'
    # args.dataset='MPI'
    args.proto='./openpose/models/pose/coco/pose_deploy_linevec.prototxt'
    args.model='./openpose/models/pose/coco/pose_iter_440000.caffemodel'
    args.dataset='COCO'
    #args.width=360
    #args.height=240
    args.width=640
    args.height=360
    #MPI
    # colors = [ [0,100,255], [50,100,255], [0,255,255], [100,100,255], [50,255,255], [150,100,255],
             # [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,150], [255,50,255],
             # [0,0,255], [255,0,0],[255,100,255], [0,0,0]]
    #COCO
    colors = [ [0,100,255], [50,100,255], [0,255,255], [100,100,255], [50,255,255], [150,100,255],
             [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,150], [255,50,255],
             [0,0,255], [255,0,0],[255,100,255],[255,100,100],[255,150,50],[200,100,50],[200,255,100],[0,0,0]]
    #MPI
    #POSE_PAIRS=[[0, 1], [1, 2], [2, 3],[3, 4], [1, 5], [5, 6],[6, 7], [1, 14],[14, 8],[8, 9], [9, 10], [14, 11],[11, 12], [12, 13]]
    #COCO
    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],[1,8], [8,9], [9,10], [1,11], [11,12], [12,13],[1,0], [0,14], [14,16], [0,15], [15,17],[2,17], [5,16] ]
    # index of pafs correspoding to the POSE_PAIRS
    # eg for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],[19,20],[21,22], [23,24], [25,26], [27,28], [29,30],[47,48], [49,50], [53,54], [51,52], [55,56],[37,38], [45,46]]

    if args.dataset == 'COCO':
        BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                       "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        # POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                       # ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                       # ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                       # ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                       # ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    elif args.dataset == 'MPI':

        BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,"Background": 15 }
        # POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                       # ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                       # ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                       # ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    else:
        assert(args.dataset == 'HAND')
        BODY_PARTS = { "Wrist": 0,
                       "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                       "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                       "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
                       "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                       "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
                     }

        POSE_PAIRS = [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                       ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                       ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                       ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                       ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                       ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                       ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                       ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                       ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                       ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]

    #videonum=fil
    #while framenum<=10:
    #args.input='.\openpose\examples\media\prueba\pruebaCam11.jpg'
    #inputname='./testVDD/imagenes/v'+str(videonum)+'_'+str(framenum)+'.jpg'
    args.input=fil

    inWidth = args.width
    inHeight = args.height
    inScale = args.scale

    net = cv.dnn.readNet(cv.samples.findFile(args.proto), cv.samples.findFile(args.model))

    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
    	hasFrame, frame = cap.read()
    	if not hasFrame:
    		cv.waitKey()
    		print('no image')
    		break

    	frameWidth = frame.shape[1]
    	frameHeight = frame.shape[0]
    	inp = cv.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
    							  (0, 0, 0), swapRB=False, crop=False)
    	net.setInput(inp)
    	out = net.forward()

    	assert(len(BODY_PARTS) <= out.shape[1])

    	detected_keypoints = []
    	keypoints_list = np.zeros((0,3))
    	keypoint_id = 0
    	for i in range(len(BODY_PARTS)):
    		# Slice heatmap of corresponding body's part.
    		probMap = out[0, i, :, :]
    		probMap = cv.resize(probMap, (inWidth, inHeight))
    		mapSmooth = cv.GaussianBlur(probMap,(3,3),0,0)
    		mapMask = np.uint8(mapSmooth>args.thr)
    		keypoints = []

    		#find the blobs
    		contours, _ = cv.findContours(mapMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    		#for each blob find the maxima
    		for cnt in contours:
    			blobMask = np.zeros(mapMask.shape)
    			blobMask = cv.fillConvexPoly(blobMask, cnt, 1)
    			maskedProbMap = mapSmooth * blobMask
    			_, maxVal, _, maxLoc = cv.minMaxLoc(maskedProbMap)
    			keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    		keypoints_with_id = []
    		for i in range(len(keypoints)):
    			keypoints_with_id.append(keypoints[i] + (keypoint_id,))
    			keypoints_list = np.vstack([keypoints_list, keypoints[i]])
    			keypoint_id += 1

    		detected_keypoints.append(keypoints_with_id)

    		# # Originally, we try to find all the local maximums. To simplify a sample
    		# # we just find a global one. However only a single pose at the same time
    		# # could be detected this way.
    		# _, conf, _, point = cv.minMaxLoc(heatMap)
    		# x = (frameWidth * point[0]) / out.shape[3]
    		# y = (frameHeight * point[1]) / out.shape[2]

    		# # Add a point if it's confidence is higher than threshold.
    		# points.append((int(x), int(y)) if conf > args.thr else None)
    	for i in range(len(BODY_PARTS)):
    		for j in range(len(detected_keypoints[i])):
    			#cv.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv.LINE_AA)
    			cv.ellipse(frame,detected_keypoints[i][j][0:2],(3, 3), 0, 0, 360, colors[i], cv.FILLED)

    	# Find valid connections between the different joints of a all persons present
    	valid_pairs = []
    	invalid_pairs = []
    	n_interp_samples = 10
    	paf_score_th = 0.1
    	conf_th = 0.7
        # loop for every POSE_PAIR
    	for k in range(len(mapIdx)):
    		# A->B constitute a limb
    		pafA = out[0, mapIdx[k][0], :, :]
    		pafB = out[0, mapIdx[k][1], :, :]
    		pafA = cv.resize(pafA, (inWidth, inHeight))
    		pafB = cv.resize(pafB, (inWidth, inHeight))
    		# Find the keypoints for the first and second limb
    		candA = detected_keypoints[POSE_PAIRS[k][0]]
    		candB = detected_keypoints[POSE_PAIRS[k][1]]
    		nA = len(candA)
    		nB = len(candB)

    		# If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

    		if( nA != 0 and nB != 0):
    			valid_pair = np.zeros((0,3))
    			for i in range(nA):
    				max_j=-1
    				maxScore = -1
    				found = 0
    				for j in range(nB):
    					# Find d_ij
    					d_ij = np.subtract(candB[j][:2], candA[i][:2])
    					norm = np.linalg.norm(d_ij)
    					if norm:
    						d_ij = d_ij / norm
    					else:
    						continue
    					# Find p(u)
    					interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
    					paf_interp = []
    					for k in range(len(interp_coord)):
    						paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
    										pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        # Find E
    					paf_scores = np.dot(paf_interp, d_ij)
    					avg_paf_score = sum(paf_scores)/len(paf_scores)

                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
    					if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
    						if avg_paf_score > maxScore:
    							max_j = j
    							maxScore = avg_paf_score
    							found = 1
                    # Append the connection to the list
    				if found:
    					valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
    			valid_pairs.append(valid_pair)
    		else: # If no keypoints are detected
    			print("No Connection : k = {}".format(k))
    			invalid_pairs.append(k)
    			valid_pairs.append([])

    	# For each detected valid pair, it assigns the joint(s) to a person
    	personwiseKeypoints = -1 * np.ones((0, 19))

    	for k in range(len(mapIdx)):
    		if k not in invalid_pairs:
    			partAs = valid_pairs[k][:,0]
    			partBs = valid_pairs[k][:,1]
    			indexA, indexB = np.array(POSE_PAIRS[k])

    			for i in range(len(valid_pairs[k])):
    				found = 0
    				person_idx = -1
    				for j in range(len(personwiseKeypoints)):
    					if personwiseKeypoints[j][indexA] == partAs[i]:
    						person_idx = j
    						found = 1
    						break

    				if found:
    					personwiseKeypoints[person_idx][indexB] = partBs[i]
    					personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
    				elif not found and k < 17:
    					row = -1 * np.ones(19)
    					row[indexA] = partAs[i]
    					row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
    					row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
    					personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    	skeleton= np.zeros((len(personwiseKeypoints),18,2))
    	for i in range(17):
    		for n in range(len(personwiseKeypoints)):
    			index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
    			if -1 in index:
    				continue
    			B = np.int32(keypoints_list[index.astype(int), 0])
    			A = np.int32(keypoints_list[index.astype(int), 1])
    			cv.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 1, cv.LINE_AA)
    			#print(str(n)+" "+str(i)+" "+str(B))
    	for i in range(18):
    		for n in range(len(personwiseKeypoints)):
    			index = personwiseKeypoints[n][i]
    			if index==-1:
    				continue
    			B = np.int32(keypoints_list[index.astype(int), 0])
    			A = np.int32(keypoints_list[index.astype(int), 1])
    			#print(B,", ",A)
    			skeleton[n][i][0]=B
    			skeleton[n][i][1]=A
    	# for n in range(len(personwiseKeypoints)):
    		# for i in range(17):
    			# print(str(n)+" "+str(i)+" "+str(skeleton[n][i][0])+" "+str(skeleton[n][i][1]))
    	t, _ = net.getPerfProfile()
    	freq = cv.getTickFrequency() / 1000
    	#cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    	#name = './cuadrosCam/' + str(counting) + '.jpg'
    	#counting = counting +1
    	name = './OpenPoseImages/'+str(videonum)+'_'+str(framenum)+'.jpg'
    	print('Creating...' + name)
    	cv.imwrite(name, frame)
    	#cv.imshow('OpenPose using OpenCV', frame)
    #print(skeleton)
    #print(detected_keypoints)
    #print(personwiseKeypoints)
    print("saving "+nam)
    np.save(nam,skeleton)
    #framenum=framenum+1
    
#Print a matrix of skeletons extracted from a frame

# print("ID |  Coord. X  |  Coord. Y")
# for pers in range(len(personwiseKeypoints)):
# 	for par in range(18):
# 		num = par+1
# 		print(str(num)+"  |  "+str(skeleton[pers][par][0])+"  |  "+str(skeleton[pers][par][1]))
# 	print("")
#print(skeleton)
