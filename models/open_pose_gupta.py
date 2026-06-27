from dataclasses import dataclass
import random
import cv2
import numpy as np

from fight_detection_pose_lstm.model_base import KeypointModel, KeypointModelArgs
from fight_detection_pose_lstm.logging import logger

"""
Based on code provided by Vikas Gupta
Gupta, Vikas. Multi-Person Pose Estimation in OpenCV using OpenPose.
https://www.learnopencv2.com/multi-person-pose-estimation-in-opencv-using-openpose
"""


@dataclass
class OpenPoseArgs(KeypointModelArgs):
	def __init__(self, model_path: str, proto_path: str):
		num_keypoints = 18
		keypoint_names = [
			"Nose",
			"Neck",
			"RShoulder",
			"RElbow",
			"RWrist",
			"LShoulder",
			"LElbow",
			"LWrist",
			"RHip",
			"RKnee",
			"RAnkle",
			"LHip",
			"LKnee",
			"LAnkle",
			"REye",
			"LEye",
			"REar",
			"LEar",
			"Background",
			]
		pairs = [
			[1, 2],
			[1, 5],
			[2, 3],
			[3, 4],
			[5, 6],
			[6, 7],
			[1, 8],
			[8, 9],
			[9, 10],
			[1, 11],
			[11, 12],
			[12, 13],
			[1, 0],
			[0, 14],
			[14, 16],
			[0, 15],
			[15, 17],
			[2, 17],
			[5, 16],
		]
		self.threshold: float = 0.1
		self.width: int = 640
		self.height: int = 360
		self.scale: float = 0.003922
		self.proto_path = proto_path
		super().__init__(
			local_path=model_path,
			num_keypoints=num_keypoints,
			keypoint_names=keypoint_names,
			pairs=pairs,
		)


class OpenPoseGuptaModel(KeypointModel):
	mapIdx = [
		[31, 32],
		[39, 40],
		[33, 34],
		[35, 36],
		[41, 42],
		[43, 44],
		[19, 20],
		[21, 22],
		[23, 24],
		[25, 26],
		[27, 28],
		[29, 30],
		[47, 48],
		[49, 50],
		[53, 54],
		[51, 52],
		[55, 56],
		[37, 38],
		[45, 46],
	]

	def __init__(self, model_args):
		super().__init__(model_args)
		self.load_model()
		logger.info(f"Loaded Open Pose model from {self.args.local_path}")

	def train(self):
		return
	
	def save_model(self):
		return

	def load_model(self):
		self.net = cv2.dnn.readNet(cv2.samples.findFile(self.args.proto_path),cv2.samples.findFile(self.args.local_path),)

	def get_keypoints(self, output) -> np.ndarray:
		detected_keypoints = []
		keypoints_list = np.zeros((0, 3))
		keypoint_id = 0
		for i in range(self.args.num_keypoints):
			# Slice heatmap of corresponding body's part.
			probMap = output[0, i, :, :]
			probMap = cv2.resize(probMap, (self.args.width, self.args.height))
			mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
			mapMask = np.uint8(mapSmooth > self.args.threshold)
			keypoints = []

			# find the blobs
			contours, _ = cv2.findContours(
				mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
			# for each blob find the maxima
			for cnt in contours:
				blobMask = np.zeros(mapMask.shape)
				blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
				maskedProbMap = mapSmooth * blobMask
				_, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
				keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
			keypoints_with_id = []
			for i in range(len(keypoints)):
				keypoints_with_id.append(keypoints[i] + (keypoint_id,))
				keypoints_list = np.vstack([keypoints_list, keypoints[i]])
				keypoint_id += 1

			detected_keypoints.append(keypoints_with_id)

			# Find valid connections between the different joints of a all persons present
		valid_pairs = []
		invalid_pairs = []
		n_interp_samples = 10
		paf_score_th = 0.1
		conf_th = 0.7
		# loop for every POSE_PAIR
		for k in range(len(self.mapIdx)):
			# A->B constitute a limb
			pafA = output[0, self.mapIdx[k][0], :, :]
			pafB = output[0, self.mapIdx[k][1], :, :]
			pafA = cv2.resize(pafA, (self.args.width, self.args.height))
			pafB = cv2.resize(pafB, (self.args.width, self.args.height))
			# Find the keypoints for the first and second limb
			candA = detected_keypoints[self.args.pairs[k][0]]
			candB = detected_keypoints[self.args.pairs[k][1]]
			nA = len(candA)
			nB = len(candB)

			# If keypoints for the joint-pair is detected
			# check every joint in candA with every joint in candB
			# Calculate the distance vector between the two joints
			# Find the PAF values at a set of interpolated points between the joints
			# Use the above formula to compute a score to mark the connection valid

			if nA != 0 and nB != 0:
				valid_pair = np.zeros((0, 3))
				for i in range(nA):
					max_j = -1
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
						interp_coord = list(
							zip(
								np.linspace(
									candA[i][0], candB[j][0], num=n_interp_samples
								),
								np.linspace(
									candA[i][1], candB[j][1], num=n_interp_samples
								),
							)
						)
						# Find L(p(u))
						paf_interp = []
						for k in range(len(interp_coord)):
							paf_interp.append(
								[
									pafA[
										int(round(interp_coord[k][1])),
										int(round(interp_coord[k][0])),
									],
									pafB[
										int(round(interp_coord[k][1])),
										int(round(interp_coord[k][0])),
									],
								]
							)
						# Find E
						paf_scores = np.dot(paf_interp, d_ij)
						avg_paf_score = sum(paf_scores) / len(paf_scores)

						# Check if the connection is valid
						# If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
						if (
							len(np.where(paf_scores > paf_score_th)[0])
							/ n_interp_samples
						) > conf_th:
							if avg_paf_score > maxScore:
								max_j = j
								maxScore = avg_paf_score
								found = 1
					# Append the connection to the list
					if found:
						valid_pair = np.append(
							valid_pair,
							[[candA[i][3], candB[max_j][3], maxScore]],
							axis=0,
						)

				# Append the detected connections to the global list
				valid_pairs.append(valid_pair)
			else:  # If no keypoints are detected
				print("No Connection : k = {}".format(k))
				invalid_pairs.append(k)
				valid_pairs.append([])

		# For each detected valid pair, it assigns the joint(s) to a person
		personwiseKeypoints = -1 * np.ones((0, 19))

		for k in range(len(self.mapIdx)):
			if k not in invalid_pairs:
				partAs = valid_pairs[k][:, 0]
				partBs = valid_pairs[k][:, 1]
				indexA, indexB = np.array(self.args.pairs[k])

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
						personwiseKeypoints[person_idx][-1] += (
							keypoints_list[partBs[i].astype(int), 2]
							+ valid_pairs[k][i][2]
						)

					# if find no partA in the subset, create a new subset
					elif not found and k < 17:
						row = -1 * np.ones(19)
						row[indexA] = partAs[i]
						row[indexB] = partBs[i]
						# add the keypoint_scores for the two keypoints and the paf_score
						row[-1] = (
							sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2])
							+ valid_pairs[k][i][2]
						)
						personwiseKeypoints = np.vstack([personwiseKeypoints, row])
		skeleton = np.zeros((len(personwiseKeypoints), 18, 2))
		for i in range(17):
			for n in range(len(personwiseKeypoints)):
				index = personwiseKeypoints[n][np.array(self.args.pairs[i])]
				if -1 in index:
					continue
				B = np.int32(keypoints_list[index.astype(int), 0])
				A = np.int32(keypoints_list[index.astype(int), 1])
		for i in range(18):
			for n in range(len(personwiseKeypoints)):
				index = personwiseKeypoints[n][i]
				if index == -1:
					continue
				B = np.int32(keypoints_list[index.astype(int), 0])
				A = np.int32(keypoints_list[index.astype(int), 1])
				# print(B,", ",A)
				skeleton[n][i][0] = B
				skeleton[n][i][1] = A
		return skeleton

	def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
		"""Draw keypoints on an image and return both original and new image

		Args:
			frame (np.ndarray): frame
			keypoints (np.ndarray): keypoints

		Returns:
			np.ndarray: drawn frame
		"""
		frame = cv2.resize(frame, (self.args.width, self.args.height))
		drawn_frame = frame.copy()
		for n in range(keypoints.shape[0]):
			color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
			for idx, pair in enumerate(self.args.pairs):
				kpt_one = (int(keypoints[n][pair[0]][0]), int(keypoints[n][pair[0]][1])) 
				kpt_two = (int(keypoints[n][pair[1]][0]), int(keypoints[n][pair[1]][1]))
				kpt_one_exists = kpt_one[0] >0 and kpt_one[1]> 0
				kpt_two_exists = kpt_two[0] > 0 and kpt_two[1] > 0
				if kpt_one_exists:
					cv2.ellipse(
						drawn_frame,
						kpt_one,
						(3, 3),
						0,
						0,
						360,
						color,
						cv2.FILLED,
					)
				if kpt_two_exists:
					cv2.ellipse(
						drawn_frame,
						kpt_two,
						(3, 3),
						0,
						0,
						360,
						color,
						cv2.FILLED,
					)
				if kpt_one_exists and kpt_two_exists:
					cv2.line(drawn_frame, kpt_one, kpt_two, color, 1, cv2.LINE_AA)
		return frame, drawn_frame

	def inference(self, x_input: np.ndarray) -> np.ndarray:
		"""Get an image as input and return keypoints of people.

		Args:
			x_input (np.ndarray): input frame

		Returns:
			np.ndarray: keypoints
		"""
		net_input = cv2.dnn.blobFromImage(
			x_input,
			self.args.scale,
			(self.args.width, self.args.height),
			(0, 0, 0),
			swapRB=False,
			crop=False,
		)
		self.net.setInput(net_input)
		logger.info("Open Pose inference starting")
		output = self.net.forward()
		logger.info("Open Pose inference finished")
		if self.args.num_keypoints <= output.shape[1]:
			keypoints = self.get_keypoints(output)
		else:
			keypoints = np.zeros((1, self.args.num_keypoints, 2))
		return keypoints
