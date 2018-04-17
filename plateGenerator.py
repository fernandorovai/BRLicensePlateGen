# This script is responsible for generating artificial brazilian plates
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import os
import sys

class PlateGenerator:
    def __init__(self, exportImage=False, exportBox=False, visualizePlates=True):
        self.dataFolder      = 'data'
        self.letters         = ["A", "B", "C", "D", "E", "F", "G",
                                "H", "I", "J", "K", "L", "M", "N",
                                "O", "P", "Q", "R", "S", "T", "U",
                                "V", "Y", "W", "X"]  # Add z
        self.numbers         = [x for x in range (0, 9)]
        self.bboxes          = []
        self.nLetters        = 3
        self.nNumbers        = 4
        self.charPadding     = 0
        self.initialWidth    = 30
        self.initialHeight   = 55
        self.widthRef        = 30
        self.heightRef       = 55

        self.exportImage     = exportImage
        self.exportBox       = exportBox
        self.visualizePlates = visualizePlates
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
        plateImgs  = []
        plateBoxes = []

        for idx in range(0, numOfPlates):
            plateSample = self.plateIm.copy()
            lettersImg  = self.generateLetters(plateSample)
            dashImg     = self.generateDash(lettersImg)
            numbersImg  = self.generateNumbers(dashImg)

            plateImgs.append(numbersImg)
            plateBoxes.append(self.bboxes)

            if self.visualizePlates:
                self.visualizePlate(numbersImg)

            self.resetReferences()
        return plateBoxes, plateImgs


if __name__ == '__main__':

    if len(sys.argv) > 1:
        numOfPlates = int(sys.argv[1])
        if numOfPlates > 0:
            plateGen = PlateGenerator()
            plateBoxes, plateImgs = plateGen.generatePlates(numOfPlates=numOfPlates)
    else:
        print("You should specify the number of plates")



