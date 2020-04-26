# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            # Non-Prius Colors
            "- red": (255, 0, 0),  # rgb(255, 0, 0)
            "- green": (0, 255, 0),  # rgb(0, 255, 0)
            "- blue": (0, 0, 255),  # rgb(0, 0, 255)
            "- white": (255, 255, 255),  # rgb(255, 255, 255)
            "- black": (0, 0, 0),  # rgb(0, 0, 0)
            "- green": (0, 255, 0),  # rgb(0, 255, 0)
            "- deep green": (0, 102, 0),  # rgb(0, 102, 0)
            "- dark gray": (76, 76, 76),  # rgb(76, 76, 76)
            "- very reddish brown": (51, 25, 25),  # rgb(51, 25, 25)
            "- medium gray": (204, 204, 204),  # rgb(204, 204, 204)
            "- light brilliant cyan": (76, 250, 255),  # rgb(76, 250, 255)
            "- strong cornflower blue": (0, 119, 178),  # rgb(0, 119, 178)
            "- vivid azure": (0, 109, 229),  # rgb(0, 109, 229)
            "- very light azure": (153, 201, 255),  # rgb(153, 201, 255)
            "- strong azure": (0, 72, 153),  # rgb(0, 72, 153)
            "- deep azure": (0, 48, 102),  # rgb(0, 48, 102)
            "- very dark azure": (25, 37, 51),  # rgb(25, 37, 51)
            "- very pale azure": (204, 227, 255),  # rgb(204, 227, 255)
            "- brilliant yellow": (230, 236, 39),  # rgb(230, 236, 39)
            "- brilliant tangelo": (236, 117, 39),  # rgb(236, 117, 39)
            "- luminous vivid tangelo": (255, 100, 0),  # rgb(255, 100, 0)
            "- strong tangelo": (153, 60, 0),  # rgb(153, 60, 0)
            "- dark sapphire blue": (19, 43, 96),  # rgb(19, 43, 96)
            "- dark grayish sapphire blue": (39, 48, 75),  # rgb(39, 48, 75)
            "- grayish sapphire blue": (80, 97, 137),  # rgb(80, 97, 137)
            "- moderate azure": (29, 88, 155),  # rgb(29, 88, 155)
            "- grayish lime green": (165, 180, 119),  # rgb(165, 180, 119)
            "- light amberish gray": (227, 225, 220),  # rgb(227, 225, 220)
            "- turquoisish gray": (136, 148, 145),  # rgb(136, 148, 145)
            "- spring green blackish": (35, 39, 37),  # rgb(35, 39, 37)
            "- cornflower bluish gray": (140, 163, 175),  # rgb(140, 163, 175)
            "- dark grayish cerulean": (64, 82, 87),  # rgb(64, 82, 87)
            "- dark phthalo blue": (30, 37, 79),  # rgb(30, 37, 79)
            "- curleanish gray": (149, 180, 190),  # rgb(149, 180, 190)
            # rgb(197, 242, 250)
            "- pale, light grayish artic blue": (197, 242, 250),
            "- light ceruleanish gray": (202, 239, 248),  # rgb(202, 239, 248)
            # rgb(156, 207, 247)
            "- pale, light grayish azure": (156, 207, 247),
            "- moderate cobalt blue": (70, 95, 143),  # rgb(70, 95, 143)

            # Prius Colors
            "+ moderate cerulean": (97, 174, 199),  # rgb(97, 174, 199)
            # rgb(152, 205, 219)
            "+ pale, light grayish cerulean": (152, 205, 219),
            # rgb(183, 228, 243)
            "+ pale, light grayish cerulean": (183, 228, 243),
            "+ grayish cerulean": (67, 112, 130),  # rgb(67, 112, 130)
            "+ grayish azure": (86, 131, 167),  # rgb(86, 131, 167)
            "+ light cornflower blue": (112, 179, 210),  # rgb(112, 179, 210)
            "+ arctic bluish gray": (142, 178, 184),  # rgb(142, 178, 184)
            "+ dark grayish arctic blue": (72, 110, 117),  # rgb(72, 110, 117)
            "+ light artic blue": (108, 200, 220),  # rgb(108, 200, 220)
            "+ moderate azure": (68, 109, 151),  # rgb(68, 109, 151)
            # rgb(70, 94, 108)
            "+ dark grayish cornflower blue": (70, 94, 108),
            "+ grayish cornflower blue": (68, 106, 34),  # rgb(68, 106, 34)
            "+ grayish cornflower blue": (92, 132, 153),  # rgb(92, 132, 153)
            "+ grayish cornflower blue": (136, 179, 199),  # rgb(136, 179, 199)
            "+ moderate cornflower blue": (41, 104, 146),  # rgb(41, 104, 146)
            "+ grayish cyan": (129, 180, 181),  # rgb(129, 180, 181)
            "+ dark cyan": (54, 125, 125),  # rgb(54, 125, 125)
        })

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
