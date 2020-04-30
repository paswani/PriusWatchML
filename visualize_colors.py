import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json


def load_json():
	print("Started Reading JSON file")
	with open("color_counts.json", "r") as read_file:
		print("Converting JSON encoded data into Python dictionary")
		counts = json.load(read_file)
	return counts
	#	print("Decoded JSON Data From File")
		#for key, value in developer.items():
		#	print(key, ":", value)
	#	print("Done reading json file")

def prep_data():
	counts = load_json()

	colors = list(counts["prius"].keys())

	# Create Labels
	for key in counts["vehicle"].keys():
		if key not in colors:
			colors.append(key)

	# Create Prius Values
	prius_values = []
	vehicle_values = []
	for color in colors:
		if color in counts["prius"].keys():
			prius_values.append(counts["prius"][color])
		else:
			prius_values.append(0)

		if color in counts["vehicle"].keys():
			vehicle_values.append(counts["vehicle"][color])
		else:
			vehicle_values.append(0)

	return colors, prius_values, vehicle_values

def grouped_bar_chart():
	labels, prius_values, vehicle_values = prep_data()

	x = np.arange(len(labels))  # the label locations
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(x - width / 2, prius_values, width, label='Prius')
	rects2 = ax.bar(x + width / 2, vehicle_values, width, label='Vehicle')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Color Counts')
	ax.set_title('Vehicles by Color')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()

	def autolabel(rects):
		"""Attach a text label above each bar in *rects*, displaying its height."""
		for rect in rects:
			height = rect.get_height()
			ax.annotate('{}'.format(height),
			            xy=(rect.get_x() + rect.get_width() / 2, height),
			            xytext=(0, 3),  # 3 points vertical offset
			            textcoords="offset points",
			            ha='center', va='bottom')
	plt.xticks(rotation=50)

	autolabel(rects1)
	autolabel(rects2)

	fig.tight_layout()

	plt.show()

grouped_bar_chart()
