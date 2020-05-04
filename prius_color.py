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
	"imprint",  # 5C7670
	"thor",  # 697F79
	"metamorphis",  # 809D96
	"boulevard",  # 8EA5A0
	"juniper [2]",  # 74918E
	"gateway",  # 5E7175
	"juniper [2]",  # 6D9292
	"desaturated cyan",  # 669999
	"steel teal",  # 5F8A8B
	"cadet [2]",  # 5F9EA0
	"wishlist",  # 659295
	"gothic [2]",  # 698890
	"jungle mist [2]",  # B0C4C4
	"streetwise",  # 4F6971
	"blue bayoux [2]",  # 62777E
	"cadet [2]",  # 536872
	"dark electric blue",  # 536878
	"wavelength",  # 3C6886
	"metallic blue",  # 4F738E
	"grayish cerulean",  # 7D9EA8
	"grayish cyan",  # 7DA8A8
	"moderate cerulean",  # 4A91A8
	"moderate azure",  # 4A79A8
	"grayish azure",  # 7D93A8
	"PMS549",  # 5E99AA
	"horizon [2]",  # 648894
	"hoki [2]",  # 647D86
	"PMS550",  # 87AFBF
	"gothic [2]",  # 6D92A1
	"greyblue",  # 77A1B5
	"bluegrey",  # 85A3B2
	"abacus",  # 768993
	"moonstone blue",  # 73A9C2
	"pewter blue",  # 8BA8B7
	"PMS5425",  # 8499A5
	"PMS5415",  # 607C8C
	"blue moon",  # 7296AB
	"nepal [2]",  # 93AAB9
	"tsunami",  # 6B8393
	"bermuda grey [2]",  # 6F8C9F
	"weldon blue",  # 7C98AB
	"hemisphere",  # 4E93BA
	"moderate cornflower blue",  # 4A85A8
	"rackley",  # 5D8AA8
	"air superiority blue",  # 72A0C1
	"moby",  # 8EB2BE
	"optimist",  # 2B688D
	"venice blue [2]",  # 2C5778
	"greyish blue",  # 5E819D
]

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


def write_json(data, filename="color_counts.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)


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
	#	print("Color: " str(key) "  Count: " str(value))

# write_json(results_dict)
