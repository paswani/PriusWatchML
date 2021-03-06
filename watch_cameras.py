from tensorflow.python.util import deprecation
import tensorflow as tf
import argparse
import json
import os

import time
import PriusImageCache
import cv2
import numpy as np
import requests
from PIL import Image
from PriusImageCache import ImageDeduplication
from yolo4 import Yolo4
from imageai.Prediction.Custom import CustomImagePrediction
from prius_color import has_prius_color_from_array
from io import BytesIO
from yolo4_model.utils import letterbox_image

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PER_MODULE_WARNING_LIMIT = 0

cams = []

cam_threads = []

dedup = ImageDeduplication()

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--action", default='detect',
                help="detect or predict")
ap.add_argument("-c", "--cams", default='min_cams2.json',
                help="Cam JSON")
ap.add_argument("-p", "--path", default='./',
                help='Image Path')
ap.add_argument("-y", "--mode", default='camera',
                help='camera or file')
ap.add_argument("-o", "--output", default='./',
                help='Output Path')
ap.add_argument("-a", "--accuracy", default=70,
                help="predict accuracy")
ap.add_argument("-z", "--predictspeed", default='normal',
                help="prediction speed")
ap.add_argument("-n", "--name", default='predictor1',
                help="name")
ap.add_argument("-m", "--models", default='./',
                help='model path')
ap.add_argument("-q", "--model", default='model_ex-006_acc-0.994420.h5',
                help='model')

args = vars(ap.parse_args())

if os.path.isdir(args['output']) is False:
    os.mkdir(args['output'])

model_path = args['models'] + 'yolo4_weight.h5'
anchors_path = args['models'] + 'yolo4_anchors.txt'
classes_path = args['models'] + 'coco_classes.txt'

score = 0.5
iou = 0.5
model_image_size = (608, 608)
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
# self.prediction.setModelPath(model_path + "model_ex-012_acc-0.988819.h5")
prediction.setModelPath(args['models'] + args['model'])
prediction.setJsonPath(args['models'] + "model_class.json")
prediction.loadModel(num_objects=2, prediction_speed=args["predictspeed"])


def write_json(data, filename=args['output'] + "prius_results.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_result(result):
    with open(args['path'] + "prius_results.json") as json_file:
        data = json.load(json_file)
        temp = data

        temp.append(result)
    write_json(data)


def predict_camera(cam, img):
    predictions, probabilities = prediction.predictImage(img,
                                                         input_type="array",
                                                         result_count=2)
    # start3 = time.time()
    now = time.localtime()
    # print("Prediction Time: " + str(start3 - start2))
    frame_time = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + "_" + str(
        now.tm_hour) + "-" + str(now.tm_min) + "_" + str(now.tm_sec)

    frame_file = frame_time + "_" + str(cam['id']) + ".jpg"
    frame_match_file = frame_time + "_match_" + str(cam['id']) + ".jpg"
    frame_detected_file = frame_time + "_detected_" + str(cam['id']) + ".jpg"

    results = []
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if "prius" in eachPrediction and int(eachProbability) > int(args['accuracy']):
            # colorStart = time.time()
            detected_color = has_prius_color_from_array(img)

            # colorEnd = time.time()
            # print("has_color Time: " + str(colorEnd - colorStart))

            if detected_color is not None:
                # print("detected color: " + detected_color)
                success = {
                    'timestamp': frame_time,
                    'image_name': frame_match_file,
                    'detected_name': frame_detected_file,
                    'image_path': args['output'],
                    'cam_id': str(cam['id']),
                    'probability': str(eachProbability),
                    'color': detected_color,
                    'predictor': args['name']
                }
                results.append(success)
    return results


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
        frame_detected_file = frame_time + \
            "_detected_" + str(cam['id']) + ".jpg"
        # frame_dir = args["path"] + frame_folder
        # if os.path.exists(frame_dir) is False:
        #	os.mkdir(frame_dir)
        img_data = []
        decoded = Image
        if "camera" in args['mode']:
            img_data = requests.get(cam['url']).content
            bytes_io = bytearray(img_data)
            decoded = Image.open(BytesIO(bytes_io))
            img_src = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)

        if "file" in args['mode']:
            decoded = Image.open(cam['path'] + cam['id'])
            img_src = cv2.imread(cam['path'] + cam['id'])
            img_data = bytes(str(img_src), 'utf8')

        if dedup.is_image_duplicate(img_data, cam['id']) is not True:
            # Update hash
            if args['mode'] is "camera":
                dedup.put_hash(img_data, cam['id'])

            # start1 = time.time()
            result = yolo4_model.detect_image(
                decoded, model_image_size=model_image_size)
            # start2 = time.time()

            # print("Detection Time: " + str(start2 - start1))
            for car in result:
                top, left, bottom, right = car["box"]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(decoded.size[1], np.floor(
                    bottom + 0.5).astype('int32'))
                right = min(decoded.size[0], np.floor(
                    right + 0.5).astype('int32'))

                img = img_src[top:bottom, left:right]

                results = predict_camera(cam, img)
                count = 0
                for success in results:
                    try:
                        r = requests.post("http://priusvision.azurewebsites.net/api/PriusTrigger",
                                          data=json.dumps(success))
                        print(
                            "---->  PRIUS IDENTIFIED.   Data: " + str(success))
                    except Exception as e:
                        print("Saving Prius result failed:  " + str(e))
                        save_result(success)

                    if r.status_code is not 200:
                        print("POST Failed.  Saving manually.")
                        save_result(success)

                        # with open(args['output'] + frame_match_file, 'wb') as handler:
                        #	handler.write(img_data)
                    decoded.save(args['output'] +
                                 str(count) + frame_match_file)
                    cv2.imwrite(args['output'] + str(count) +
                                "_detected_" + frame_detected_file, img)
                    count = count + 1
    except Exception as e:
    	pass


if __name__ == '__main__':

    if "file" in args['mode']:
        print("File Mode")
        for root, dirs, files in os.walk(args["path"]):
            for name in files:
                if "predict" in args['action']:
                    img = cv2.imread(root + "/" + name)
                    results = predict_camera(dict(id=name, path=root), img)
                    for found in results:
                        cv2.imwrite(args['output'] + "_detected_" + name +
                                    str(int(found['probability'])), img)
                        print(str(found))
                else:
                    if "processed" not in name and "detection" not in name and name.endswith(".jpg"):
                        watch_camera(dict(id=name, path=root))

    if "camera" in args['mode']:
        print("Loading Seattle Cams")
        with open(args["cams"], "r") as read_file:
            cams = json.load(read_file)

        while True:
            for cam in cams:
                watch_camera(cam)
