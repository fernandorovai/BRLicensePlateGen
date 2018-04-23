# This script is responsible for generating artificial brazilian plates
# and performing augmentation (consider real data styles).
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
import time
import matplotlib.pyplot as plt
import random
import os
import sys
import collections
import imgaug as ia
import numpy as np

class PlateGenerator:
    def __init__(self, showPlates=True, showStatistics=True, augmentation=True):
        self.dataFolder      = 'data'
        self.letters         = ["A", "B", "C", "D", "E", "F", "G",
                                "H", "I", "J", "K", "L", "M", "N",
                                "O", "P", "Q", "R", "S", "T", "U",
                                "V", "Y", "W", "X", "Z"]

        self.statistics      = collections.OrderedDict([("A",0), ("B",0), ("C",0), ("D",0), ("E",0),
                                                        ("F",0), ("G",0), ("H",0), ("I",0), ("J",0),
                                                        ("K",0), ("L",0), ("M",0), ("N",0), ("O",0),
                                                        ("P",0), ("Q",0), ("R",0), ("S",0), ("T",0),
                                                        ("U",0), ("V",0), ("Y",0), ("W",0), ("X",0),
                                                        ("Z",0), ("0",0), ("1",0), ("2",0), ("3",0),
                                                        ("4",0), ("5",0), ("6",0), ("7",0), ("8",0),
                                                        ("9",0), ("-",0)])
        self.numbers         = [x for x in range (0, 10)]
        self.bboxes          = []
        self.nLetters        = 3
        self.nNumbers        = 4
        self.charPadding     = 0
        self.initialWidth    = 30
        self.initialHeight   = 55
        self.widthRef        = 30
        self.heightRef       = 55

        self.showStatistics  = showStatistics
        self.visualizePlates = showPlates
        self.augmentation    = augmentation
        self.plateSample     = os.path.join(self.dataFolder, 'plateSample01.jpg')
        self.plateIm         = Image.open(self.plateSample)

        self.resetReferences()

    def resetReferences(self):
        self.widthRef  = self.initialWidth
        self.heightRef = self.initialHeight
        self.bboxes    = []

    def generateLetters(self, image):
        # Adding letters
        for _ in range(0, self.nLetters):
            randomChar = random.choice(self.letters)
            char = Image.open(os.path.join(self.dataFolder, "%s.png" % str(randomChar)))
            charW, charH = char.size

            # Append box according to widthRef + charW
            self.bboxes.append(self.generateBox(charW, charH, str(randomChar)))

            image.paste(char, (self.widthRef, self.heightRef), char)
            self.widthRef += charW + self.charPadding

            # Increment statistics
            self.statistics[randomChar] +=1
        return image

    def generateBox(self, charW, charH, tag):
        xMin = self.widthRef
        yMin = self.heightRef
        xMax = self.widthRef + charW + self.charPadding
        yMax = self.heightRef + charH
        return xMin,yMin, xMax, yMax, tag

    def generateNumbers(self, image):
        # Adding numbers
        for _ in range(0, self.nNumbers):
            randomNum = random.choice(self.numbers)
            number = Image.open(os.path.join(self.dataFolder, "%s.png" % str(randomNum)))
            numberW, numberH = number.size

            # Append box according to widthRef + numberW
            self.bboxes.append(self.generateBox(numberW, numberH, randomNum))

            image.paste(number, (self.widthRef, self.heightRef), number)
            self.widthRef += numberW + self.charPadding

            # Increment statistics
            self.statistics[str(randomNum)] +=1
        return image

    def generateDash(self, image):
        # Adding dash
        dash = Image.open(os.path.join(self.dataFolder, "%s.png" % str("-")))
        dashW, dashH = dash.size

        # Append box according to widthRef + numberW
        self.bboxes.append(self.generateBox(dashW, dashH, "-"))

        image.paste(dash, (self.widthRef, self.heightRef), dash)
        self.widthRef += dashW + self.charPadding

        # Increment statistics
        self.statistics["-"] += 1
        return image

    def visualizePlate(self, image):
        draw = ImageDraw.Draw(image)
        for box in self.bboxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], None, (0,255,0))
            draw.rectangle([(box[0]-1, box[1]-1), (box[2]+1, box[3]+1)], None, (0,255,0))
        plt.imshow(image)
        plt.show()

    def generatePlates(self, numOfPlates, testSet=False):
        print("------------------------------------------------------------------")
        print("Generating Artficial Data...")
        startTime = time.time()

        plates     = []

        for idx in range(0, numOfPlates):
            plateSample = self.plateIm.copy()
            lettersImg  = self.generateLetters(plateSample)
            dashImg     = self.generateDash(lettersImg)
            finalImg  = self.generateNumbers(dashImg)

            plates.append({"plateIdx": idx, "plateImg": finalImg, "plateBoxes": self.bboxes})

            # Visualize plate
            if self.visualizePlates:
                self.visualizePlate(finalImg)

            # Reset references (width, height and boxes)
            self.resetReferences()

        # Show histogram
        if self.showStatistics:
            self.visualizeStatistics()

        # Perform data augmentation
        if self.augmentation:
            if testSet: plates = self.augmentImgsTest(plates)
            else:       plates = self.augmentImgs(plates)

        elapsed = round((time.time() - startTime),3)

        print("Plates generated succesfully in %s seconds" % str(elapsed))
        return plates

    def visualizeStatistics(self):
        plt.figure()
        plt.title("Characters Histogram")
        plt.bar(self.statistics.keys(), self.statistics.values(), 1, color='g')
        plt.show()

    def augmentImgsTest(self, plates):
        augPlates = []
        for plate in plates:
            plateIdx = plate["plateIdx"]
            plateImg = np.asarray(plate['plateImg'])
            plateBoxes = plate['plateBoxes']
            bbs = []
            seq = iaa.Sequential([
                iaa.Sometimes(0.9, iaa.GaussianBlur(sigma=(0, 0.7))),
                iaa.ContrastNormalization((0.75, 2.0)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2 * 255), per_channel=0.6),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Sometimes(0.8, iaa.Affine(rotate=(-10, 15), shear=(-8, 8))),
                iaa.Sometimes(0.8, iaa.Affine(rotate=(-5, 7), shear=(-13, 15))),
                iaa.Sometimes(0.8, iaa.Affine(scale=(0.7, 0.7))),
                iaa.Sometimes(0.8, iaa.Affine(shear=(-3, 3)))], random_order=True)
            seq_det = seq.to_deterministic()

            for box in plateBoxes:
                bbs.append(ia.BoundingBox(box[0], box[1], box[2], box[3]))

            bbsOnImage = ia.BoundingBoxesOnImage(bbs, shape=plateImg.shape)
            imageAug = seq_det.augment_images([plateImg])[0]
            bboxAug = seq_det.augment_bounding_boxes([bbsOnImage])[0]
            bboxAug = bboxAug.remove_out_of_image().cut_out_of_image()

            finalImg = Image.fromarray(imageAug)
            bboxAugFormatted = []
            for idx, box in enumerate(bboxAug.bounding_boxes):
                bboxAugFormatted.append((box.x1, box.y1, box.x2, box.y2, plateBoxes[idx][4]))

            augPlates.append(
                {"plateIdx": plateIdx, "plateImg": Image.fromarray(imageAug), "plateBoxes": bboxAugFormatted})

            # Visualize plate
            if self.visualizePlates:
                self.visualizePlate(finalImg)

        return augPlates


    def augmentImgs(self, plates):
        augPlates = []
        for plate in plates:
            plateIdx    = plate["plateIdx"]
            plateImg    = np.asarray(plate['plateImg'])
            plateBoxes  = plate['plateBoxes']
            bbs         = []
            seq         = iaa.Sequential([
                          iaa.Sometimes(0.9,iaa.GaussianBlur(sigma=(0, 0.7))),
                          iaa.ContrastNormalization((0.75, 1.9)),
                          iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.6),
                          iaa.Multiply((0.8, 1.2), per_channel=0.2),
                          iaa.Sometimes(0.8, iaa.Affine(rotate=(-5, 5), shear=(-8, 8))),
                          iaa.Sometimes(0.8, iaa.Affine(rotate=(-7, 7), shear=(-3, 3))),
                          iaa.Sometimes(0.8, iaa.Affine(scale=(0.6, 0.6))),
                iaa.Sometimes(0.8, iaa.Affine(shear=(-3, 3)))], random_order=True)
            seq_det     = seq.to_deterministic()

            for box in plateBoxes:
                bbs.append(ia.BoundingBox(box[0], box[1], box[2], box[3]))

            bbsOnImage  = ia.BoundingBoxesOnImage(bbs,shape=plateImg.shape)
            imageAug    = seq_det.augment_images([plateImg])[0]
            bboxAug     = seq_det.augment_bounding_boxes([bbsOnImage])[0]
            bboxAug     = bboxAug.remove_out_of_image().cut_out_of_image()

            finalImg    = Image.fromarray(imageAug)
            bboxAugFormatted = []
            for idx, box in enumerate(bboxAug.bounding_boxes):
                bboxAugFormatted.append((box.x1, box.y1, box.x2, box.y2, plateBoxes[idx][4]))

            augPlates.append({"plateIdx": plateIdx, "plateImg": Image.fromarray(imageAug), "plateBoxes": bboxAugFormatted})

            # Visualize plate
            if self.visualizePlates:
                self.visualizePlate(finalImg)

        return augPlates

    def getStatistics(self):
        return self.statistics

if __name__ == '__main__':
    if len(sys.argv) > 1:
        numOfPlates = int(sys.argv[1])
        if numOfPlates > 0:
            plateGen = PlateGenerator(showPlates=True)
            plates = plateGen.generatePlates(numOfPlates=numOfPlates)
    else:
        print("You should specify the number of plates")



