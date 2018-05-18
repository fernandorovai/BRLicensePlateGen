# Code developed by Flavio (AI2BIZ) and adapted by Fernando Rodrigues Jr
import cv2
import numpy as np
import configs.extractor_config as extractorCfg
import os
from PIL import Image, ImageDraw
import collections
import matplotlib.pyplot as plt

maxCharWidth         = extractorCfg.CONFIGS['maxCharWidthFactor']
minCharWidth         = extractorCfg.CONFIGS['minCharWidthFactor']
maxCharHeight        = extractorCfg.CONFIGS['maxCharHeightFactor']
minCharHeight        = extractorCfg.CONFIGS['minCharHeightFactor']
pathToTrainImageDir  = extractorCfg.CONFIGS['trainImageDir']
imageType            = extractorCfg.CONFIGS['imageType']

class RealPlateExtractor:

    def __init__(self):
        self.statistics = collections.OrderedDict([("A", 0), ("B", 0), ("C", 0), ("D", 0), ("E", 0),
                                                   ("F", 0), ("G", 0), ("H", 0), ("I", 0), ("J", 0),
                                                   ("K", 0), ("L", 0), ("M", 0), ("N", 0), ("O", 0),
                                                   ("P", 0), ("Q", 0), ("R", 0), ("S", 0), ("T", 0),
                                                   ("U", 0), ("V", 0), ("Y", 0), ("W", 0), ("X", 0),
                                                   ("Z", 0), ("0", 0), ("1", 0), ("2", 0), ("3", 0),
                                                   ("4", 0), ("5", 0), ("6", 0), ("7", 0), ("8", 0),
                                                   ("9", 0), ("-", 0)])

    # Morphological test for contours at license plates
    def validContour(self, maxH, minH, maxW, minW, w, h):
        bValue = True
        if w > maxW:
            bValue = False
        if h > maxH:
            bValue = False
        if w < minW:
            bValue = False
        if h < minH:
            bValue = False
        return bValue


    def enhance(self, img):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
        return cv2.filter2D(img, -1, kernel)

    # Do char segmentation
    def segmentChars(self, loadedImg, basename):
        # Load the radar image
        img = loadedImg
        height, width = img.shape[:2]
        counter = 0
        boxes = []

        # Create metrics
        expected_max_height = maxCharHeight * height
        expected_min_height = minCharHeight * height
        expected_max_width  = maxCharWidth * width
        expected_min_width  = minCharWidth * width

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # Apply adaptive threshold with Otsu
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the contours
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, find the bounding rectangle and draw it
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self.validContour(expected_max_height, expected_min_height, expected_max_width, expected_min_width, w, h):
                xMin = x
                yMin = y
                xMax = x + w
                yMax = y + h
                tag  = basename[counter]
                boxes.append((xMin, yMin, xMax, yMax, tag))
                self.statistics[tag] +=1
                counter +=1
        return boxes

    def getStatistics(self):
        return self.statistics

    def extractBoxesFromImage(self, showPlates=False):
        trainImages = [os.path.join(pathToTrainImageDir, file) for file in os.listdir(pathToTrainImageDir) if
                       file.endswith(imageType)]

        plates = []
        imgId  = 0
        for image in trainImages:
            basename = (os.path.basename(image).split("."))[0]
            basename = list(basename)
            basename.remove("-")

            loadedImg = cv2.imread(image)
            loadedImg = cv2.cvtColor(loadedImg, cv2.COLOR_BGR2RGB)
            boxes = self.segmentChars(loadedImg, basename)
            plates.append({"plateIdx": imgId, "plateImg": Image.fromarray(loadedImg), "plateBoxes": boxes})
            imgId += 1

            if showPlates:
                self.visualizePlate(Image.fromarray(loadedImg), boxes)

        return plates

    def visualizePlate(self, image, boxes):
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], None, (0,255,0))
            draw.rectangle([(box[0]-1, box[1]-1), (box[2]+1, box[3]+1)], None, (0,255,0))
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    realPlateExtractor = RealPlateExtractor()
    plates = realPlateExtractor.extractBoxesFromImage()