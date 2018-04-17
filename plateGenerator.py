# This script is responsible for generating artificial brazilian plates
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import os
import sys
import collections

class PlateGenerator:
    def __init__(self, exportImage=False, exportBox=False, showPlates=True, showStatistics=True):
        self.dataFolder      = 'data'
        self.letters         = ["A", "B", "C", "D", "E", "F", "G",
                                "H", "I", "J", "K", "L", "M", "N",
                                "O", "P", "Q", "R", "S", "T", "U",
                                "V", "Y", "W", "X"]  # Add z

        self.statistics      = collections.OrderedDict([("A",0), ("B",0), ("C",0), ("D",0), ("E",0),
                                                        ("F",0), ("G",0), ("H",0), ("I",0), ("J",0),
                                                        ("K",0), ("L",0), ("M",0), ("N",0), ("O",0),
                                                        ("P",0), ("Q",0), ("R",0), ("S",0), ("T",0),
                                                        ("U",0), ("V",0), ("Y",0), ("W",0), ("X",0),
                                                        ("0",0), ("1",0), ("2",0), ("3",0), ("4",0),
                                                        ("5",0), ("6",0), ("7",0), ("8",0), ("9",0)])
        self.numbers                = [x for x in range (0, 9)]
        self.bboxes                 = []
        self.nLetters               = 3
        self.nNumbers               = 4
        self.charPadding            = 0
        self.initialWidth           = 30
        self.initialHeight          = 55
        self.widthRef               = 30
        self.heightRef              = 55

        self.exportImage            = exportImage
        self.exportBox              = exportBox
        self.showStatistics         = showStatistics
        self.visualizePlates        = showPlates
        self.plateSample            = os.path.join(self.dataFolder, 'plateSample01.jpg')
        self.plateIm                = Image.open(self.plateSample)

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

            #increment statistics
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

            #increment statistics
            self.statistics[str(randomNum)] +=1
        return image

    def generateDash(self, image):
        # Adding dash
        dash = Image.open(os.path.join(self.dataFolder, "%s.png" % str("dash")))
        dashW, dashH = dash.size

        image.paste(dash, (self.widthRef, self.heightRef), dash)
        self.widthRef += dashW + self.charPadding
        return image

    def visualizePlate(self, image):
        draw = ImageDraw.Draw(image)
        for box in self.bboxes:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], None, (0,255,0))
        plt.imshow(image)
        plt.show()

    def generatePlates(self, numOfPlates):
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

        return plates

    def visualizeStatistics(self):
        plt.figure()
        plt.title("Characters Histogram")
        plt.bar(self.statistics.keys(), self.statistics.values(), 1, color='g')
        plt.show()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        numOfPlates = int(sys.argv[1])
        if numOfPlates > 0:
            plateGen = PlateGenerator(showPlates=True)
            plates = plateGen.generatePlates(numOfPlates=numOfPlates)
    else:
        print("You should specify the number of plates")



