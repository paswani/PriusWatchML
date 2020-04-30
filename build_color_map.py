import re
import os
from PIL import ImageColor
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def write_json(data, filename="colors.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)

def save_result(entry):
	with open("colors.json") as json_file:
		data = json.load(json_file)
		d = data

		data.append(entry)

	write_json(data)

def structure_colors_data():
	color_text = open("colors_unstructured.txt", "rt")  # open lorem.txt for reading text
	contents = color_text.read()  # read the entire file into a string
	color_text.close()  # close the file
	#print(contents)  # print contents

	colors = []

	for color in re.finditer(r"((?:(?:\w|\')+\s){0,5}?(?:(?:\w+\s){1,4}|(?:\[\d{1,5}\])\s))(\#.{6})", contents):
		color_entry = {
			"color": color.group(1).strip(),
			"hex": color.group(2),
			"rgb": ImageColor.getcolor(color.group(2), "RGB")
		}
		save_result(color_entry)

def load_json():
	print("Started Reading JSON file")
	with open("colors.json", "r") as read_file:
		print("Converting JSON encoded data into Python dictionary")
		developer = json.load(read_file)

		print("Decoded JSON Data From File")
		for key, value in developer.items():
			print(key, ":", value)
		print("Done reading json file")
