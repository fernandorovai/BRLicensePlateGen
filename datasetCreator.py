#This script generates dataset based on a specific framework structure
import io
import os
import matplotlib.pyplot as plt
from plateGenerator import PlateGenerator
from TFRecordWriter import TFRecordWriter, TFExample
from time import time
from imgBBoxExtractor import RealPlateExtractor
import numpy as np

class DatasetCreator:
    def __init__(self, numOfPlates, showPlates=False, balanceData=False,
                 showStatistics=False, augmentation=True, trainSet=True,
                 lbFile = False, includeDash=False, realData=False,
                 resize=False, model=0, split=True, outputPath=os.getcwd(),
                 bgInsertion=False, contourOnly=True):

        if realData:
            plateGen = RealPlateExtractor()
            self.plates = plateGen.extractBoxesFromImage(showPlates)
        else:
            plateGen    = PlateGenerator(showPlates=showPlates, augmentation=augmentation, bgInsertion=bgInsertion)
            self.plates = plateGen.generatePlates(numOfPlates=numOfPlates, trainSet=trainSet, includeDash=includeDash, resize=resize)

        self.balanceData        = balanceData
        self.showStatistics     = showStatistics
        self.labelFile          = lbFile
        self.contourOnly        = contourOnly
        self.includeDash        = includeDash
        self.classes = {"plate": 1}
        if not self.contourOnly:
            self.classes            = { "A": 1, "B": 2, "C":  3,  "D": 4, "E": 5, "F": 6,
                                        "G": 7, "H": 8, "I1": 9,  "J":10, "K":11, "L":12,
                                        "M":13, "N":14, "O0":15,  "P":16, "Q":17, "R":18,
                                        "S":19, "T":20, "U": 21,  "V":22, "Y":23, "W":24,
                                        "X":25, "Z":26, "2": 27,  "3":28, "4":29, "5":30,
                                        "6":31, "7":32, "8": 33,  "9":34, "-":35}

        statistics             = plateGen.getStatistics()
        self.maxCharOccurrence = min(val for val in statistics.values() if val > 0)
        self.occurrenceControl = statistics.fromkeys(statistics, 1)


        if model == 0:
            if split:
                train, validation = np.split(self.plates, [int(.8 * len(self.plates))])
                tfRecordTrainFilename = "%s_train.tfrecord" % output
                tfRecordTestFilename = "%s_test.tfrecord" % output

                self.createTensorFlowDataset(train, tfRecordTrainFilename)
                self.createTensorFlowDataset(validation, tfRecordTestFilename)
            else:
                tfRecordTrainFilename = "%s_train.tfrecord" % output
                self.createTensorFlowDataset(self.plates, tfRecordTrainFilename)

        elif model == 1:
            self.createYOLOV2Dataset()
        else:
            print("Model not found")


    def createYOLOV2Dataset(self):
        # To be defined
        print("This feature is under development")

    def createTensorFlowDataset(self, plates, tfRecordFilename):
        tfLabelMapFilename = "%s_label_map.pbtxt" % output
        startTime = time()
        print("------------------------------------------------------------------")
        print("Generating TensorFlow Dataset with (%d) license plates" % len(plates))
        tfRecordGen = TFRecordWriter(tfRecordFilename)
        diffClasses = []

        for idx, plate in enumerate(plates):
            plateIdx         = plate['plateIdx']
            plateImg         = plate['plateImg']
            plateBoxes       = plate['plateBoxes']
            byteStream       = io.BytesIO()
            plateImg.save(byteStream, 'jpeg')
            imageBytes       = byteStream.getvalue()
            height           = plateImg.height
            width            = plateImg.width
            xMins            = []  # List of normalized left x coordinates in bounding box (1 per box)
            xMaxs            = []  # List of normalized right x coordinates in bounding box (1 per box)
            yMins            = []  # List of normalized top y coordinates in bounding box (1 per box)
            yMaxs            = []  # List of normalized bottom y coordinates in bounding box (1 per box)
            classesText      = []  # List of string class name of bounding box (1 per box)
            classes          = []  # List of integer class id of bounding box (1 per box)
            encodedImageData = imageBytes
            imageFormat      = b'jpeg'
            groundTruth      = ''

            for box in plateBoxes:

                xMin = box[0]
                yMin = box[1]
                xMax = box[2]
                yMax = box[3]
                char = box[4]

                if not self.contourOnly:
                    groundTruth += str(char)

                # increment occurrence counter
                self.occurrenceControl[str(char)] +=1

                if int(self.occurrenceControl[str(char)]) > int(self.maxCharOccurrence) and self.balanceData == True:
                    continue

                if char == "1" or char == "I":
                    char = "I1"
                elif char == "0" or char == "O":
                    char = "O0"

                xMins.append(float(xMin) / float(width))
                yMins.append(float(yMin) / float(height))
                xMaxs.append(float(xMax) / float(width))
                yMaxs.append(float(yMax) / float(height))
                classesText.append(str(char).encode('utf-8'))
                classes.append(self.classes[str(char)])
                if not any(el['classID'] == self.classes[str(char)] for el in diffClasses):
                    diffClasses.append({"classID":self.classes[str(char)] , "className": str(char)})

            # Avoid empty plates
            if len(classes) == 0:
                continue


            if self.contourOnly:
                groundTruth = "plate_%s" % (str(idx))

            # Append data to TFRecord
            tfRecordExample                  = TFExample()
            tfRecordExample.width            = width
            tfRecordExample.height           = height
            tfRecordExample.filename         = ("%s" % groundTruth).encode('utf-8')
            tfRecordExample.sourceID         = (str(plateIdx).zfill(7)).encode('utf-8')
            tfRecordExample.encodedImageData = encodedImageData
            tfRecordExample.imageFormat      = imageFormat
            tfRecordExample.xMins            = xMins
            tfRecordExample.xMaxs            = xMaxs
            tfRecordExample.yMins            = yMins
            tfRecordExample.yMaxs            = yMaxs
            tfRecordExample.classesText      = classesText
            tfRecordExample.classes          = classes

            tfExample = tfRecordGen.createTfExample(tfRecordExample)
            tfRecordGen.appendExampleToTfStream(tfExample)

        # Create pbtxt if specified
        if self.labelFile:
            # sort label map and append to the pbtxt file
            diffClasses = sorted(diffClasses, key=lambda d: d['classID'], reverse=False)
            self.createTFLabelMap(diffClasses, tfLabelMapFilename)

        tfRecordGen.closeTfStream()
        elapsed = round((time() - startTime),3)
        print("TensorFlow dataset created successfully! - %s - Process took %s seconds" % (str(tfRecordFilename), str(elapsed)))
        if self.showStatistics:
            self.visualizeStatistics()

    def createTFLabelMap(self, diffClasses, outputPath):
        file = open(outputPath, 'a+')
        for cls in diffClasses:
            file.write("item { id: %d \n name: \"%s\" }\n" % (cls['classID'], str(cls['className'])))
        file.close()

    def visualizeStatistics(self):
        plt.figure()
        plt.title("Characters Histogram")
        plt.bar(self.occurrenceControl.keys(), self.occurrenceControl.values(), 1, color='r')
        plt.show()


if __name__ == '__main__':
    numOfPlates, model, balanced, dash, showPlates, \
    augmentation, trainSet, output, split, bgInsertion, \
    contourOnly = 0, 0, False, False, False, False, True, "", False, True, True

    path             = os.getcwd()
    realData         = input("Want to use real data? (y/n):")
    output           = input("What is the set name? ")
    model            = int(input("0 - Tensorflow \n1 - YOLOV2\nWhat is the model? (e.g: 0 or 1): "))
    resize           = input("Want to resize the image? (y/n): ")
    if realData == ('n' or 'N'):
        numOfPlates  = int(input("How many plates do you want to generate? \nNumber of plates:"))
        split        = input("Want to split dataset into train-test? (y/n): ")
        bgInsertion  = input("Want to add a random background? (y/n): ")
        augmentation = input("Want to augment the dataset? (y/n): ")
        dash         = input("Want to include the dash between letters and numbers? (y/n): ")
        balanced     = input("Want to balance the data? (You may have images with few annotations) (y/n): ")
        # trainSet     = input("Is it a train set(y) or test set(n)? (y/n): ")
    lblFile          = input("Want to generate the label pbtxt file? (y/n): ")
    showPlates       = input("Want to see generated plates? (y/n): ")

    if (int(numOfPlates) > 0 or realData == ('y' or 'Y')) and (model == 0 or model == 1) and output != "":
        output = os.path.join(path, output)

        if realData == ('y' or 'Y'): realData = True
        else: realData = False

        if balanced == ('y' or 'Y'): balanced = True
        else: balanced = False

        if showPlates == ('y' or 'Y'): showPlates = True
        else: showPlates = False

        if bgInsertion == ('y' or 'Y'): bgInsertion = True
        else: bgInsertion = False

        if augmentation == ('y' or 'Y'): augmentation = True
        else: augmentation = False

        # if trainSet == ('y' or 'Y'): trainSet = True
        # else: trainSet = False

        if lblFile == ('y' or 'Y'): lblFile = True
        else: lblFile = False

        if dash == ('y' or 'Y'): dash = True
        else: dash = False

        if resize == ('y' or 'Y'): resize = True
        else: resize = False

        if split == ('y' or 'Y'): split = True
        else: split = False

        # if not trainSet:
        #     output = output + 'Test'

        # Create train set
        DatasetCreator(numOfPlates, showPlates=showPlates, balanceData=balanced,
                        trainSet=trainSet, augmentation=augmentation, lbFile=lblFile,
                        realData=realData, resize=resize, model=model, split=split,
                        outputPath=output, contourOnly=contourOnly, bgInsertion=bgInsertion)
    else:
        print("Sorry, you chose something that does not match the requirements!")