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
	"juniper [2]",  #74918E
	"gateway",  #5E7175
	"juniper [2]",  #6D9292
	"desaturated cyan",  #669999
	"steel teal",  #5F8A8B
	"cadet [2]",  #5F9EA0
	"wishlist",  #659295
	"gothic [2]",  #698890
	"streetwise",  #4F6971
	"blue bayoux [2]",  #62777E
	"cadet [2]",  #536872
	"dark electric blue",  #536878
	"wavelength",  #3C6886
	"metallic blue",  #4F738E
	"grayish cerulean",  #7D9EA8
	"grayish cyan",  #7DA8A8
	"moderate cerulean",  #4A91A8
	"moderate azure",  #4A79A8
	"PMS549",  #5E99AA
	"horizon [2]",  #648894
	"hoki [2]",  #647D86
	"PMS550",  #87AFBF
	"gothic [2]",  #6D92A1
	"greyblue",  #77A1B5
	"bluegrey",  #85A3B2
	"moonstone blue",  #73A9C2
	"pewter blue",  #8BA8B7
	"PMS5415",  #607C8C
	"blue moon",  #7296AB
	"tsunami",  #6B8393
	"hemisphere",  #4E93BA
	"moderate cornflower blue",  #4A85A8
	"rackley",  #5D8AA8
	"air superiority blue",  #72A0C1
	"moby",  #8EB2BE
	"optimist",  #2B688D
	"venice blue [2]",  #2C5778
	"greyish blue",  #5E819D
	"surfie green [2]", ##0C7A79
	"ming [2]", ##407577
	"mosque [2]", ##036A6E
	"muse", ##2E696D
	"paradiso [2]", ##317D82
	"retro", ##1B5256
	"deep teal [3]", ##00555A
	"PMS3165", ##00565B
	"PMS5473", ##26686D
	"deep aqua", ##08787F
	"william [2]", ##3A686C
	"casal [2]", ##2F6168
	"petrol", ##005F6A
	"PMS315", ##006B77
	"blue lagoon [3]", ##017987
	"metallic seaweed", ##0A7E8C
	"deep arctic blue", ##004E59
	"dauntless", ##166F7F
	"maestro", ##005E6D
	"boomtown", ##346672
	"kitsch", ##006C7F
	"breaker bay [2]", ##5DA19F
	"cyan [5]", ##008B8B
	"dark cyan [3]", ##008B8B
	"strong cyan", ##00A8A8
	"patriot", ##4F9292
	"java [2]", ##259797
	"moderate cyan", ##4AA8A8
	"blue chill [2]", ##408F90
	"viridian green [3]", ##009698
	"PMS5483", ##609191
	"juniper [2]", ##6D9292
	"PMS320", ##009EA0
	"desaturated cyan", ##669999
	"cadet blue [10]", ##5F9F9F
	"grayish cyan", ##7DA8A8
	"PMS321", ##008789
	"steel teal", ##5F8A8B
	"bounce", ##679394
	"cadet [2]", ##5F9EA0
	"turquoise [8]", ##00868B
	"kumutoto", ##78AFB2
	"half baked [2]", ##558F93
	"wishlist", ##659295
	"neptune [2]", ##77A8AB
	"fountain blue [2]", ##65ADB2
	"paradiso [2]", ##488084
	"PMS3145", ##00848E
	"ziggurat [2]", ##81A6AA
	"hullabaloo", ##008B97
	"yabbadabbadoo", ##008B97
	"such fun", ##489EA8
	"PMS3125", ##00B7C6
	"retreat", ##39909B
	"gumbo [2]", ##7CA1A6
	"PMS3135", ##009BAA
	"moderate arctic blue", ##4A9CA8
	"turquoise blue [5]", ##06B1C4
	"PMS631", ##54B7C6
	"cerulean [5]", ##05B8CC
	"seeker", ##0092A5
	"wot eva", ##4495A4
	"pelorous [2]", ##3EABBF
	"strong arctic blue", ##0093A8
	"scooter [2]", ##308EA0
	"eastern blue [2]", ##1E9AB0
	"PMS312", ##00ADC6
	"glacier [2]", ##78B1BF
	"toto", ##519DAF
	"PMS632", ##00A0BA
	"teal blue [4]", ##01889F
	"viking [2]", ##4DB1C8
	"onepoto", ##81D3D1
	"PMS318", ##93DDDB
	"java [2]", ##1FC2C2
	"robin's egg blue [2]", ##00CCCC
	"robin egg blue [4]", ##00CCCC
	"cyan [5]", ##00CDCD
	"vivid cyan", ##00E7E7
	"dark slate grey [5]", ##79CDCD
	"medium turquoise [3]", ##70DBDB
	"brilliant cyan", ##51E7E7
	"pale turquoise [7]", ##96CDCD
	"PMS629", ##B2D8D8
	"light cyan [6]", ##8BE7E7
	"pale light grayish cyan", ##B8E7E7
	"turquoise [8]", ##ADEAEA
	"dark turquoise [5]", ##00CED1
	"turquoise dark", ##00CED1
	"PMS319", ##4CCED1
	"bright light blue", ##26F7FD
	"morning glory [2]", ##9EDEE0
	"half kumutoto", ##9CC8CA
	"PMS3105", ##7FD6DB
	"PMS2975", ##BAE0E2
	"neptune [2]", ##7CB7BB
	"sea serpent", ##4BC7CF
	"aquamarine [9]", ##78DBE2
	"aquamarine blue", ##71D9E2
	"PMS304", ##A5DDE2
	"turquoise blue [5]", ##77DDE7
	"cadet blue [10]", ##7AC5CD
	"PMS630", ##8CCCD3
	"aqua blue", ##02D8E9
	"PMS636", ##99D6DD
	"PMS3115", ##2DC6D6
	"half baked [2]", ##85C4CC
	"PMS310", ##72D1DD
	"powder blue [4]", ##B0E0E6
	"light arctic blue", ##8BDCE7
	"robin's egg", ##6DEDFD
	"spray [2]", ##79DEEC
	"viking [2]", ##64CCDB
	"brilliant arctic blue", ##51D5E7
	"blizzard blue [2]", ##A3E3ED
	"blue lagoon [3]", ##ACE5EE
	"light brilliant arctic blue", ##65ECFF
	"PMS311", ##28C4D8
	"charlotte [2]", ##A4DCE6
	"scooter [2]", ##2EBFD4
	"PMS637", ##6BC9DB
	"medium sky blue", ##80DAEB
	"sky blue [9]", ##80DAEB
	"PMS305", ##70CEE2
	"vivid arctic blue", ##00CAE7
	"luminous vivid arctic blue", ##00DFFF
	"refresh", ##71B8CA
	"non photo blue", ##A4DDED
	"meltwater", ##6EAEC0
	"hippie blue [2]", ##49889A
	"PMS313", ##0099B5
	"PMS549", ##5E99AA
	"awash", ##739CA9
	"PMS638", ##00B5D6
	"PMS314", ##00829B
	"blue [12]", ##0093AF
	"smalt blue [2]", ##51808F
	"tax break [2]", ##51808F
	"horizon [2]", ##648894
	"pacific blue [2]", ##1CA9C9
	"blue green [3]", ##199EBD
	"aquarius", ##43A8C5
	"PMS801", ##00AACC
	"ball blue", ##21ABCD
	"moderate cerulean", ##4A91A8
	"bondi blue [3]", ##0095B6
	"PMS639", ##00A0C4
	"gothic [2]", ##6D92A1
	"endorphin", ##4190AD
	"bright cerulean", ##1DACD6
	"cerulean [5]", ##1DACD6
	"dirty blue", ##3F829D
	"PMS640", ##008CB2
	"boston blue [2]", ##438EAC
	"greyblue", ##77A1B5
	"bluegrey", ##85A3B2
	"PMS801 2X", ##0089AF
	"abacus", ##768993
	"peacock", ##33A1C9
	"moonstone blue", ##73A9C2
	"PMS306 2X", ##00A3D1
	"shakespeare [2]", ##4EABD1
	"summer sky", ##38B0DE
	"cyan [5]", ##00B7EB
	"bowie", ##0084AC
	"waterfront", ##3E7F9D
	"blue moon", ##7296AB
	"deep sky blue [5]", ##009ACD
	"PMS2995", ##00A5DB
	"vivid cerulean [2]", ##00AEE7
	"tsunami", ##6B8393
	"hemisphere", ##4E93BA
	"PMS299", ##00A3DD
	"moderate cornflower blue", ##4A85A8
	"freefall", ##1D95C9
	"PMS2915", ##60AFDD
	"picton blue [2]", ##45B1E8
	"wedgewood [2]", ##4E7F9E
	"air force blue [2]", ##5D8AA8
	"rackley", ##5D8AA8
	"sky blue [9]", ##3299CC
	"air superiority blue", ##72A0C1
	"brilliant cornflower blue", ##51AFE7
	"ocean", ##017B92
	"allports [2]", ##1F6A7D
	"PMS633", ##007F99
	"teal blue [4]", ##367588
	"norwester", ##48798A
	"bismark [2]", ##486C7A
	"marathon", ##305563
	"PMS634", ##00667F
	"jelly bean [3]", ##44798E
	"PMS3025", ##00546B
	"undercurrent", ##365C6C
	"PMS308", ##00607C
	"calypso [2]", ##3D7188
	"sea blue [2]", ##047495
	"lucifer", ##2E5060
	"blumine [2]", ##305C71
	"astral [2]", ##376F89
	"blue sapphire", ##126180
	"deep sky blue [5]", ##00688B
	"strong cerulean", ##007EA8
	"arapawa [2]", ##274A5D
	"chathams blue [2]", ##2C5971
	"PMS307", ##007AA5
	"PMS641", ##007AA5
	"cg blue", ##007AA5
	"PMS302", ##004F6D
	"orient [2]", ##255B77
	"PMS5405", ##3F6075
	"steel blue [8]", ##236B8E
	"celadon blue", ##007BA7
	"cerulean [5]", ##007BA7
	"deep cerulean [2]", ##007BA7
	"ocean blue [2]", ##03719C
	"yeehaa", ##006C98
	"sky blue [9]", ##4A708B
	"wavelength", ##3C6886
	"PMS3015", ##00709E
	"metallic blue", ##4F738E
	"neon blue [2]", ##04D9FF
	"french pass [2]", ##A4D2E0
	"glacier [2]", ##80B3C4
	"parachute", ##65B8D1
	"very light cerulean", ##9EE7FF
	"PMS306", ##00BCE2
	"light cerulean", ##8BD0E7
	"winter wizard", ##A0E6FF
	"anakiwa [2]", ##9DE5FF
	"fresh air", ##A6E7FF
	"PMS2985", ##51BFE2
	"brilliant cerulean", ##51C2E7
	"light brilliant cerulean", ##65D8FF
	"seagull [2]", ##77B7D0
	"PMS297", ##82C6E2
	"sky blue [9]", ##87CEEB
	"bright sky blue", ##02CCFE
	"vivid sky blue", ##00CCFF
	"dark sky blue [2]", ##8CBED6
	"PMS2905", ##93C6E0
	"baby blue [3]", ##89CFF0
	"pale cyan [3]", ##87D3F8
	"PMS298", ##51B5E0
	"cornflower [6]", ##93CCEA
	"light cornflower blue [2]", ##93CCEA
	"light sky blue [6]", ##8DB6CD
	"malibu [2]", ##66B7E1
	"spiro disco ball", ##0FC0FC
	"very light cornflower blue", ##9EDBFF
	"capri [3]", ##00BFFF
	"deep sky blue [5]", ##00BFFF
	"luminous vivid cerulean", ##00BFFF
	"sky blue deep", ##00BFFF
	"lightblue", ##7BC8F6
	"sky blue light", ##87CEFA
	"light brilliant cornflower blue", ##65C5FF
	"PMS292", ##75B2DD
	"sky", ##82CAFC
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
	contours, hierarchy = cv2.findContours(
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
	topThree = []
	for i in range(0, min(len(contoursWithArea)-1, 2)):
		topThree.append(contoursWithArea[i][0])

	return topThree

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

		contours = []

		for c in cnts:
			color = cl.label(lab, c)
			contours.append(color)

		return contours
	except Exception as e:
		print("While detecting color: " + str(e))


def has_prius_color(image):
	detected_color = detect_color(image)
	if detected_color in prius_colors:
		return True
	return False

def has_prius_color_from_array(image):
	# Top 3 contours
	detected_colors = detect_color_from_array(image)
	for color in detected_colors:
		if color in prius_colors:
			return color
	return None


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
