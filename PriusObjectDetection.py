import multiprocessing
import os
import queue
import threading
import time
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
from PIL import Image
from PriusImage import PriusImage
from PriusPalette import PriusPalette
from imageai.Detection import ObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction


class PriusPredictor(object):
	def __init__(self, image_path, model_path, output_path):
		self.avgColor = []
		self.pcaColors = []

		self.detector = ObjectDetection()
		self.detector.setModelTypeAsYOLOv3()
		self.detector.setModelPath(model_path + "yolo.h5")
		self.detector.loadModel()

		self.prediction = CustomImagePrediction()
		self.prediction.setModelTypeAsResNet()
		self.prediction.setModelPath(model_path + "model_ex-027_acc-0.992647.h5")
		self.prediction.setJsonPath(model_path + "model_class.json")
		self.prediction.loadModel(num_objects=2)

		now = time.localtime()
		self.frame_folder = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday)
		self.image_path = image_path
		self.output_path = output_path + "detection/" + self.frame_folder + "/"

		if os.path.exists(image_path) is False:
			os.mkdir(image_path)

		if os.path.exists(output_path) is False:
			os.mkdir(output_path)

		if os.path.exists(os.path.join(output_path, 'detection')) is False:
			os.mkdir(os.path.join(output_path, 'detection'))

		if os.path.exists(os.path.join(output_path, 'processed')) is False:
			os.mkdir(os.path.join(output_path, 'processed'))

		self.create_output_folder()

	def create_output_folder(self):
		if os.path.exists(self.output_path) is False:
			os.mkdir(self.output_path)

	def write_prediction_results(self):
		if self.hasPcaMatch is True:
			cv2.imwrite("pca_" + file, image)
			self.pcaPredictedCount = pcaPredictedCount + 1

		if self.hasPerfectMatch is True:
			cv2.imwrite("perfect_" + file, image)
			self.perfectMatchCount = perfectMatchCount + 1

	def predict_vehicle(self, prediction_meta):
		detected_img = prediction_meta['image_path']
		if self.detect_pca(detected_img):
			print("PCA match for: " + detected_img)

		return self.prediction.predictImage(detected_img, result_count=2)


	def detect_pca(self, image):
		priusImage = PriusImage.from_path(image)
		return priusImage.has_pca_match()

	def detect_vehicle(self, meta_data):
		try:

			image = meta_data["image_path"] + meta_data['image_name']
			output_image = self.output_path + meta_data['image_name']
			print("Detecting vehicle for " + image + " -> " + output_image)

			if os.path.exists(image) is not True:
				print("File doesnt exist. File: " + "/frames/" + image)
			detections, objects_path = self.detector.detectObjectsFromImage(input_image=image,
			                                                           extract_detected_objects=True,
			                                                           output_image_path=output_image,
			                                                           minimum_percentage_probability=50)

			return zip(detections, objects_path)
		except Exception as e:
			print("While detecting vehicle: " + str(e))