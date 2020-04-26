# import the necessary packages
import argparse
import os

import cv2
import imutils
from ColorLabeler import ColorLabeler
from ShapeDetector import ShapeDetector
from imutils import contours
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
                help="path to the input image")
ap.add_argument("-i", "--image", required=False,
                help="path to the input image")
args = vars(ap.parse_args())

path = args['path']
arr = os.listdir(path)

'''
[
   {
      "prius": [
         {
            "image_name": "car1",
            "image_path": "../imgpath",
            "contours": [
               {
                  "red": 0,
                  "green": 0,
                  "blue": 0,
                  "shade": "cornflower",
                  "coords": 0
               }
            ]
         }
      ],
      "not_prius": [
         {
            "image_name": "car1",
            "image_path": "../imgpath",
            "contours": [
               {
                  "red": 0,
                  "green": 0,
                  "blue": 0,
                  "shade": "cornflower",
                  "coords": 0
               }
            ]
         }
      ]
   }
]
'''

def write_json(data, filename="colors.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)

	'''
						[{
							"red": result['red'],
							"green": result['green'],
							"blue": result['blue'],
							"shade": result['shade'],
							"coords": result['coords']
						}]
					'''


def save_result(result, file):
	with open("colors.json") as json_file:
		data = json.load(json_file)
		temp = data
		img_class = result['class']
		# python object to be appended
		y = {
			img_class: [
				{
					"image_name": result['image_name'],
					"image_path": result['image_shade'],
					"contours": result['contours']

				}
			]
		}

		temp.img_class.append(y)

	write_json(data)

def find_significant_contour(img):
	image, contours, hierarchy = cv2.findContours(
		img,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)

	# Find level 1 contours
	level1Meta = []
	for contourIndex, tupl in enumerate(hierarchy[0]):
		# Each array is in format (Next, Prev, First child, Parent)
		# Filter the ones without parent
		if tupl[3] == -1:
			tupl = np.insert(tupl.copy(), 0, [contourIndex])
			level1Meta.append(tupl)

	# From among them, find the contours with large surface area.
	contoursWithArea = []
	for tupl in level1Meta:
		contourIndex = tupl[0]
		contour = contours[contourIndex]
		area = cv2.contourArea(contour)
		contoursWithArea.append([contour, area, contourIndex])

	contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
	largestContour = contoursWithArea[0][0]
	return largestContour

def detect_color(image_src, image_name):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	image = cv2.imread(image_src)

	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])
	# blur the resized image slightly, then convert it to both
	# grayscale and the L*a*b* color spaces
	blurred = cv2.GaussianBlur(resized, (5, 5), 0)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
	thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image
	cnts = find_significant_contour(thresh.copy())

	#cnts = imutils.grab_contours(cnts)
	# initialize the shape detector and color labeler
	cl = ColorLabeler()

	contours = []
	# loop over the contours
	#for c in cnts:
	color = cl.label(lab, cnts)
	contours.append(color)
	print(str("Image " + image_name + " has contour with color " + str(color)))

	return contours


for image in arr:
	color = detect_color(os.path.join(path, image), image)
	#print(color)

'''
		contour = {
			"points": c,
			""
		}
'''
