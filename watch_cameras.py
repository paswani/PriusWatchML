import argparse
import json
import os
import queue
import re
import threading
import time
from datetime import timedelta

import PriusImageCache
import cv2
import numpy as np
import requests
from PIL import Image
from PriusImageCache import ImageDeduplication
from imageai.Detection import ObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction
from prius_color import has_prius_color_from_array
from prius_color import has_prius_contour_from_array
from timeloop import Timeloop

tl = Timeloop()
cams = []

cam_threads = []

dedup = ImageDeduplication()
q = queue.Queue()
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--timer", required=False,
                help="path to input image")
ap.add_argument("-c", "--cams", default='min_cams2.json',
                help="Cam JSON")
ap.add_argument("-p", "--path", default='./',
                help='Results Path')
ap.add_argument("-a", "--accuracy", default=50,
                help="predict accuracy")
ap.add_argument("-y", "--detectspeed", default='normal',
                help="detection speed")
ap.add_argument("-z", "--predictspeed", default='normal',
                help="prediction speed")
ap.add_argument("-n", "--name", default='predictor1',
                help="name")
ap.add_argument("-m", "--models", default='./',
                help='model path')

args = vars(ap.parse_args())

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(args['models'] + "yolo.h5")
detector.loadModel(detection_speed=args["detectspeed"])
custom_objects = detector.CustomObjects(car=True)

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
# self.prediction.setModelPath(model_path + "model_ex-012_acc-0.988819.h5")
prediction.setModelPath(args['models'] + "model_ex-043_acc-0.996787.h5")
prediction.setJsonPath(args['models'] + "model_class.json")
prediction.loadModel(num_objects=2, prediction_speed=args["predictspeed"])


def write_json(data, filename=args['path'] + "prius_results.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)


def save_result(result):
	with open(args['path'] + "prius_results.json") as json_file:
		data = json.load(json_file)
		temp = data

		temp.append(result)
	write_json(data)


def watch_camera(cam):
	#	while True:
	try:
		# cam = q.get()
		# if cam is None:
		#	break

		now = time.localtime()
		# frame_folder = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + "_" + str(
		#	now.tm_hour) + "-" + str(now.tm_min) + "/"

		frame_time = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + "_" + str(
			now.tm_hour) + "-" + str(now.tm_min) + "_" + str(now.tm_sec)

		frame_file = frame_time + "_" + str(cam['id']) + ".jpg"
		frame_match_file = frame_time + "_match_" + str(cam['id']) + ".jpg"
		frame_detected_file = frame_time + "_detected_" + str(cam['id']) + ".jpg"

		# frame_dir = args["path"] + frame_folder
		# if os.path.exists(frame_dir) is False:
		#	os.mkdir(frame_dir)

		img_data = requests.get(cam['url']).content

		if dedup.is_image_duplicate(img_data, cam['id']):
			print(frame_file + " is a duplicate image.  Removing.")
		else:
			# Update hash
			dedup.put_hash(img_data, cam['id'])
			decoded = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)

			start1 = time.time()
			result = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
			                                               input_type="array",
			                                               extract_detected_objects=True,
			                                               input_image=decoded,
			                                               output_type="array",
			                                               thread_safe=True,
			                                               minimum_percentage_probability=50)
			start2 = time.time()

			print("Detection Time: " + str(start2 - start1))

			for arr in result[1]:
				(x1, y1, x2, y2) = arr["box_points"]
				img = decoded[y1:y2, x1:x2]

				start2 = time.time()
				predictions, probabilities = prediction.predictImage(img,
				                                                     thread_safe=True,
				                                                     input_type="array",
				                                                     result_count=2)
				start3 = time.time()

				print("Prediction Time: " + str(start3 - start2))
				for eachPrediction, eachProbability in zip(predictions, probabilities):
					if "prius" in eachPrediction and int(eachProbability) > int(args['accuracy']):
						colorStart = time.time()
						has_color = has_prius_color_from_array(img)
						colorEnd = time.time()
						print("has_color Time: " + str(colorEnd - colorStart))

						if has_color is True:
							success = {
								'timestamp': frame_time,
								'image_name': frame_match_file,
								'detected_name': frame_detected_file,
								'image_path': args['path'],
								'cam_id': str(cam['id']),
								'probability': str(eachProbability),
								'predictor': args['name']
							}
							r = ''
							try:
								r = requests.post("http://priusvision.azurewebsites.net/api/PriusTrigger",
								                  data=json.dumps(success))
								print("---->  PRIUS IDENTIFIED.   Data: " + str(success))
							except Exception as e:
								print("Saving Prius result failed:  " + str(e))
								save_result(success)

							if r.status_code is not 200:
								print("POST Failed.  Saving manually.")
								save_result(success)

							with open(args['path'] + frame_match_file, 'wb') as handler:
								handler.write(img_data)

							cv2.imwrite(args['path'] + frame_detected_file, img)
	except Exception as e:
		print(e)


@tl.job(interval=timedelta(seconds=int(args["timer"])))
def watch_cameras_timer():
	print("Cameras job current time : {}".format(time.ctime()))
	for cam in cams:
		q.put(cam)


def start_watching():
	procs = 1
	for i in range(0, procs):
		process = threading.Thread(target=watch_camera)
		cam_threads.append(process)

	for t in cam_threads:
		t.start()


if __name__ == '__main__':
	print("Loading Seattle Cams")
	with open(args["cams"], "r") as read_file:
		cams = json.load(read_file)

	while True:
		for cam in cams:
			watch_camera(cam)
# q.put(cam)

# pathFolder = args['path']

# tl.start()
# start_watching()
