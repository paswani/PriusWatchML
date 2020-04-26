from collections import OrderedDict

import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist


class PriusPalette:
	def __init__(self):
		colors = OrderedDict({
			# Non-Prius Colors
			"- red": (255, 0, 0),
			"- green": (0, 255, 0),
			"- blue": (0, 0, 255),
			"- white": (255, 255, 255),
			"- black": (0, 0, 0),
			"- green": (0, 255, 0),
			"- deep green": (0, 102, 0),
			"- dark gray": (76, 76, 76),
			"- very reddish brown": (51, 25, 25),
			"- medium gray": (204, 204, 204),
			"- light brilliant cyan": (76, 250, 255),
			"- strong cornflower blue": (0, 119, 178),
			"- vivid azure": (0, 109, 229),
			"- very light azure": (153, 201, 255),
			"- strong azure": (0, 72, 153),
			"- deep azure": (0, 48, 102),
			"- very dark azure": (25, 37, 51),
			"- very pale azure": (204, 227, 255),
			"- brilliant yellow": (230, 236, 39),
			"- brilliant tangelo": (236, 117, 39),
			"- luminous vivid tangelo": (255, 100, 0),
			"- strong tangelo": (153, 60, 0),
			"- dark sapphire blue": (19, 43, 96),
			"- dark grayish sapphire blue": (39, 48, 75),
			"- grayish sapphire blue": (80, 97, 137),
			"- moderate azure": (29, 88, 155),
			"- grayish lime green": (165, 180, 119),
			"- light amberish gray": (227, 225, 220),
			"- turquoisish gray": (136, 148, 145),
			"- spring green blackish": (35, 39, 37),
			"- cornflower bluish gray": (140, 163, 175),
			"- dark grayish cerulean": (64, 82, 87),
			"- dark phthalo blue": (30, 37, 79),
			"- curleanish gray": (149, 180, 190),
			"- pale, light grayish artic blue": (197, 242, 250),
			"- light ceruleanish gray": (202, 239, 248),
			"- pale, light grayish azure": (156, 207, 247),
			"- moderate cobalt blue": (70, 95, 143),
			# Prius Colors
			"+ moderate cerulean": (97,174,199),
			"+ pale, light grayish cerulean": (152, 205, 219),
			"+ pale, light grayish cerulean": (183, 228, 243),
			"+ grayish cerulean": (67, 112, 130),
			"+ grayish azure": (86, 131, 167),
			"+ light cornflower blue": (112, 179, 210),
			"+ arctic bluish gray": (142, 178, 184),
			"+ dark grayish arctic blue": (72, 110, 117),
			"+ light artic blue": (108, 200, 220),
			"+ moderate azure": (68,109,151),
			"+ dark grayish cornflower blue": (70,94,108),
			"+ grayish cornflower blue": (68,106,34),
			"+ grayish cornflower blue": (92, 132, 153),
			"+ grayish cornflower blue": (136, 179, 199),
			"+ moderate cornflower blue": (41,104,146),
			"+ grayish cyan": (129,180,181),
			"+ dark cyan": (54,125,125),
		})

		self.requiredShades = []

		# 8db1b7 <-> #50757f -- rgb(50,87,77) <-> rgb(181,227,223)
		self.dullBlueGray = [(50, 87, 77), (181, 227, 223)]
		# +++Required Palettes: 353  Average Color: 17  PCA Colors: 70  Perfect Match: 4  Total: 30
		# self.requiredShades.append(self.dullBlueGray)

		# d3e8eb <-> #84bac8 -- rgb(133,186,200) <-> rgb(222,232,235)
		self.lightBlueShades = [(133, 186, 200), (235, 232, 222)]
		# +++Required Palettes: 333  Average Color: 0  PCA Colors: 57  Perfect Match: 0  Total: 25/31
		#self.requiredShades.append(self.lightBlueShades)

		# rgb(111,142,193)
		# 51707b - rgb(81,112,123) -- rgb(71,102,113) <-> rgb(91,122,143)
		self.shade4 = [(113, 102, 71), (193, 142, 111)]
		#   -> Required Palettes: 25  Average Color: 16  PCA Colors: 0  Perfect Match: 0  Total: 30
		# self.requiredShades.append(self.shade4)

		# 5a6676 - rgb(90,102,118) -- rgb(90,102,118) <-> rgb(150,172,178)
		self.shade6 = [(118, 102, 90), (178, 172, 150)]
		# +++Required Palettes: 329  Average Color: 10  PCA Colors: 0  Perfect Match: 0  Total: 30/31
		# self.requiredShades.append(self.shade6)

		# 345359 - rgb(52,83,89) -- rgb(32,63,69) <-> rgb(82,103,109)
		self.blue_green_shade = [(69, 63, 32), (109, 103, 72)]
		# +++Required Palettes: 354  Average Color: 26  PCA Colors: 68  Perfect Match: 14  Total: 30/31 / 29/33
		self.requiredShades.append(self.blue_green_shade)

		# 649ca0 - rgb(100,156,160) -- rgb(80,136,140) <-> rgb(120,176,180)
		self.shade12 = [(140, 136, 80), (180, 176, 120)]
		# +++Required Palettes: 81  Average Color: 0  PCA Colors: 1  Perfect Match: 0  Total: 29
		# self.requiredShades.append(self.shade12)
		# badbd5 - rgb(186,219,213) -- rgb(176,209,203) <-> rgb(196,229,223)
		self.shade5 = [(203, 209, 176), (223, 229, 196)]
		# +++Required Palettes: 245  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27/31
		# self.requiredShades.append(self.shade5)
		# d1f7f7 - rgb(209,247,247) -- rgb(199,237,237) <-> rgb(219,255,255)
		#  self.shade14 = [(237,237,199), (255,255,219)]
		# +++Required Palettes: 118  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27

		# 7b99b6 - rgb(123,153,182) -- rgb(113,143,172) <-> rgb(133,163,192)
		self.shade7 = [(172, 143, 113), (192, 163, 133)]
		# +++Required Palettes: 114  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 30/31
		# self.requiredShades.append(self.shade7)

		# c7e3ec - rgb(199,227,236) -- rgb(189,217,226) <-> rgb(209,237,246)
		self.shade8 = [(226, 217, 189), (246, 237, 209)]
		# +++Required Palettes: 198  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27/31

		# 9baeb7 - rgb(155,174,183) -- rgb(145,164,173) <-> rgb(165,184,193)
		self.shade9 = [(173, 164, 145), (193, 184, 165)]
		# +++Required Palettes: 286  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 30/31
		# self.requiredShades.append(self.shade9)

		# allocate memory for the L*a*b* image, then initialize
		# the color names list


		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []

		# loop over the colors dictionary
		for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
			self.lab[i] = rgb
			self.colorNames.append(name)

		# convert the L*a*b* array from the RGB color space
		# to L*a*b*
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def get_color(self, meta_data):
		# load the image and resize it to a smaller factor so that
		# the shapes can be approximated better
		image = cv2.imread(meta_data["image"])
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

		# loop over the contours
		for c in cnts:
			# compute the center of the contour
			M = cv2.moments(c)
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			# detect the shape of the contour and label the color
			shape = sd.detect(c)
			color = cl.label(lab, c)
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape and labeled
			# color on the image
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			text = "{} {}".format(color, shape)
			cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
			cv2.putText(image, text, (cX, cY),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			# show the output image
			cv2.imshow("Image", image)
			cv2.waitKey(0)


	def label(self, image, c):
		# construct a mask for the contour, then compute the
		# average L*a*b* value for the masked region
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]

		# initialize the minimum distance found thus far
		minDist = (np.inf, None)

		# loop over the known L*a*b* color values
		for (i, row) in enumerate(self.lab):
			# compute the distance between the current L*a*b*
			# color value and the mean of the image
			d = dist.euclidean(row[0], mean)

			# if the distance is smaller than the current distance,
			# then update the bookkeeping variable
			if d < minDist[0]:
				minDist = (d, i)

		# return the name of the color with the smallest distance
		return self.colorNames[minDist[1]]

	def required_shades(self):
		return self.requiredShades

	def has_shade(self, shade):
		outputList = []
		for boundary in self.requiredShades:
			for (lower, upper) in boundary:
				lower = np.array(lower, dtype="uint8")
				upper = np.array(upper, dtype="uint8")
				aboveLower = shade[0] >= lower[0] and shade[1] >= lower[1] and shade[2] >= lower[2]
				aboveUpper = shade[0] <= upper[0] and shade[1] <= upper[1] and shade[2] <= upper[2]
				# print("Shade: " + str(shade) + "  Palette Range: " + str(boundary) + "  Match? " + str(aboveLower and aboveUpper))
				outputList.append(aboveLower and aboveUpper)

		return all(outputList)
