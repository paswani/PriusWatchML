# import the necessary packages
import argparse
import json
import os

import cv2
import imutils
import numpy as np
from ColorLabeler import ColorLabeler
from imutils import contours

# construct the argument parse and parse the arguments
prius_list = [
"aurometalsaurus",
"bermuda grey [2]",
"bali hai [2]",
"tsunami",
"abacus",
"sirocco [2]",
"venice blue [2]",
"wishlist",
"tapa [2]",
"old silver",
"regent grey [2]",
"metallic blue",
"steel teal",
"cadet [2]",
"imprint",
"greyblue",
"wavelength",
"greyish blue",
"grayish cerulean",
"gateway",
"PMS550",
"scotty silver",
"moderate cornflower blue",
"PMS5425",
"elevate",
"PMS549",
"air superiority blue",
"grayish azure",
"gothic [2]",
"forecast",
"hemisphere",
"hoki [2]",
"pewter blue",
"blue bayoux [2]",
"moby",
"washed green",
"nepal [2]",
"boulevard",
"bluff",
"weldon blue",
"blue moon",
"horizon [2]",
"crescent",
"obelisk",
"metamorphis",
"compass",
"escapade",
"desaturated cyan",
"PMS5415",
"light slate grey",
"optimist",
"bluegrey",
"juniper [2]",
"innocence",
"streetwise",
"moonstone blue",
"anemone green",
"steel",
"oslo grey [2]",
"go ben [2]",
"hit grey [2]",
"jungle mist [2]",
"instinct",
"thor",
"shuttle grey [2]",
"metro",
"dark electric blue",
"nevada [2]",
"smoke",
"quarter tuna",
"triple delta",
"avatar",
"double delta",
"meridian",
"storm grey [2]",
"rackley"]

prius_colors = frozenset(prius_list)
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
                help="path to the input image")
ap.add_argument("-i", "--image", required=False,
                help="path to the input image")
ap.add_argument("-t", "--type", required=False,
                help="Prius or Vehicle")
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

'''
def write_json(data, filename="color_counts.json"):
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
'''


def find_significant_contour(img):
	image,contours, hierarchy = cv2.findContours(
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
	return contoursWithArea[0][0]

def get_contour_colors(image_src):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	return get_contour_colors_from_array(cv2.imread(image_src))

def get_contour_colors_from_array(image):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	try:
		resized = imutils.resize(image, width=300)
		ratio = image.shape[0] / float(resized.shape[0])
		# blur the resized image slightly, then convert it to both
		# grayscale and the L*a*b* color spaces
		blurred = cv2.GaussianBlur(resized, (5, 5), 0)
		gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
		lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

		# find contours in the thresholded image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		cl = ColorLabeler()

		contours = []
		# loop over the contours
		for c in cnts:
			color = cl.label(lab, c)
			contours.append(color)
		return contours
	except Exception as e:
		print("While getting contours: " + str(e))

def has_prius_contour(image):
	colors = get_contour_colors(image)
	for color in colors:
		if color in prius_colors:
			return True
	return False

def has_prius_contour_from_array(arr):
	colors = get_contour_colors_from_array(arr)
	for color in colors:
		if color in prius_colors:
			return True
	return False

def detect_color(image_src):
	return detect_color_from_array(cv2.imread(image_src))

def detect_color_from_array(image):
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	# image = cv2.imread(os.path.join(path,image))
	try:
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

		cl = ColorLabeler()
		color = cl.label(lab, cnts)
		return color
	except Exception as e:
		print("While detecting color: " + str(e))

def has_prius_color(image):
	detected_color = detect_color(image)
	if detected_color in prius_colors:
		return True
	return False

def has_prius_color_from_array(image):
	detected_color = detect_color_from_array(image)
	if detected_color in prius_colors:
		return True
	return False

def load_counts():
	with open("color_counts.json", "r") as read_file:
		return json.load(read_file)

def detect_colors():
	results_dict = load_counts()
	results = {}
	if args['type'] == "prius":
		results = results_dict['prius']
	else:
		results = results_dict['vehicle']

	for image in arr:
		color = detect_color(os.path.join(path, image), image)
		if color is not None:
			if color in results:
				count = results[color]
				results[color] = count + 1
			else:
				results[color] = 1

	# print(str(results_dict))
	# for key, value in results_dict['prius']:
	#	print("Color: " + str(key) + "  Count: " + str(value))

#write_json(results_dict)
