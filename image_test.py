import time
import json
import cv2
import numpy as np
import requests
from PIL import Image
from imageai.Detection import ObjectDetection
from imageai.Prediction.Custom import CustomImagePrediction
from prius_color import has_prius_color_from_array

def write_json(data, filename= "prius_results.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)



def save_result(result):
	with open("prius_results.json") as json_file:
		data = json.load(json_file)
		temp = data

		temp.append(result)
	write_json(data)



detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./yolo.h5")
detector.loadModel(detection_speed="fastest")

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
# self.prediction.setModelPath(model_path + "model_ex-012_acc-0.988819.h5")
prediction.setModelPath("./model_ex-043_acc-0.996787.h5")
prediction.setJsonPath("./model_class.json")
prediction.loadModel(prediction_speed="fastest")

custom_objects = detector.CustomObjects(car=True)

accuracy = 0
imgs = []

data = requests.get("http://seattle.gov/trafficcams/images/15_NW_65_NS.jpg").content
decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
imgs.append(decoded)
data = requests.get("http://seattle.gov/trafficcams/images/15_NW_65_NS.jpg").content
decoded = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
imgs.append(decoded)

for decoded in imgs:

	start1 = time.time()
	result = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
	                                               input_type="array",
	                                               extract_detected_objects=True,
	                                               input_image=np.array(decoded),
	                                               output_type="array",
	                                               minimum_percentage_probability=50)
	start2 = time.time()
	print("Detection Time: " + str(start2 - start1))
	detected = []
	for arr in result[1]:
		(x1, y1, x2, y2) = arr["box_points"]
		img = decoded[y1:y2, x1:x2]

		colorStart = time.time()
		has_color = has_prius_color_from_array(img)
		colorEnd = time.time()
		print("has_color Time: " + str(colorEnd - colorStart))

		if has_color is not True:
			start2 = time.time()
			predictions, probabilities = prediction.predictImage(img, input_type="array", result_count=2)
			start3 = time.time()

			print("Prediction Time: " + str(start3 - start2))
			for eachPrediction, eachProbability in zip(predictions, probabilities):
				if "prius" in eachPrediction and eachProbability > accuracy:
					try:
						success = {
							#'timestamp': frame_time,
						#	'image_name': frame_file,
						#	'image_path': frame_dir,
						#	'cam_id': str(cam['id']),
							'probability': str(eachProbability)
							#'predictor':  'series'
						}
						r = requests.post("http://priusvision.azurewebsites.net/api/PriusTrigger", data=json.dumps(success))
						print("---->  PRIUS IDENTIFIED.   Data: " + str(success))
					except Exception as e:
						print("Saving Prius result failed:  " + str(e))
						save_result(success)

					if r.status_code is not 200:
						print("POST Failed.  Saving manually.")
						save_result(success)

					with open("./test/img2.jpg", 'wb') as handler:
						handler.write(data)

					cv2.imwrite("./test/img1.jpg", img)







