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
		self.prediction.setModelPath(model_path + "model_ex-012_acc-0.988819.h5")
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

	def predict_vehicle(self, prediction_meta):
		detected_img = os.path.join(prediction_meta['image_path'], prediction_meta['image_name'])
		#if self.detect_pca(detected_img):
		#	print("PCA match for: " + detected_img)

		return self.prediction.predictImage(detected_img, result_count=2)


	def detect_pca(self, image):
		priusImage = PriusImage.from_path(image)
		return priusImage.has_pca_match()

	def detect_vehicle(self, meta_data):
		try:

			image = os.path.join(meta_data["image_path"], meta_data['image_name'])
			output_image = self.output_path + meta_data['image_name']
			print("Detecting vehicle for " + meta_data['image_name'] + " -> " + output_image)

			if os.path.exists(image) is not True:
				print("File doesnt exist. File: "  + image)
			custom_objects = detector.CustomObjects(car=True)
			detections, objects_path = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
			                                                                 input_image=image,
			                                                                 extract_detected_objects=True,
			                                                                 output_image_path=output_image,
			                                                                 minimum_percentage_probability=50)

			return zip(detections, objects_path)
		except Exception as e:
			print("While detecting vehicle: " + str(e))