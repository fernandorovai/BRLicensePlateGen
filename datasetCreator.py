#This script generates dataset based on a specific framework structure
import io
from plateGenerator import PlateGenerator
from TFRecordWriter import TFRecordWriter, TFExample
from time import time
import matplotlib.pyplot as plt

class DatasetCreator:
    def __init__(self, numOfPlates, showPlates=False, balanceData=False, showStatistics=False, augmentation=True, testSet=False):
        plateGen            = PlateGenerator(showPlates=showPlates, augmentation=augmentation)
        self.balanceData    = balanceData
        self.showStatistics = showStatistics
        self.plates         = plateGen.generatePlates(numOfPlates=numOfPlates, testSet=testSet)
        self.classes        = { "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,
                                "G": 7, "H": 8, "I": 9, "J":10, "K":11, "L":12,
                                "M":13, "N":14, "O":15, "P":16, "Q":17, "R":18,
                                "S":19, "T":20, "U":21, "V":22, "Y":23, "W":24,
                                "X":25, "Z":26, "0":27, "1":28, "2":29, "3":30,
                                "4":31, "5":32, "6":33, "7":34, "8":35, "9":36,
                                "-":37}
        statistics = plateGen.getStatistics()
        self.maxCharOccurrence = min(val for val in statistics.values() if val > 0)
        self.occurrenceControl = statistics.fromkeys(statistics, 1)

    def createYOLOV2Dataset(self):
        # To be defined
        print("This feature is under development")

    def createTensorFlowDataset(self, tfRecordPath):
        tfRecordFilename   = "%s.tfrecord" % output
        tfLabelMapFilename = "%s_label_map.pbtxt" % output
        startTime = time()
        print("------------------------------------------------------------------")
        print("Generating TensorFlow Dataset with (%d) license plates" % len(self.plates))
        tfRecordGen = TFRecordWriter(tfRecordFilename)
        diffClasses = []

        for plate in self.plates:
            plateIdx         = plate['plateIdx']
            plateImg         = plate['plateImg']
            plateBoxes       = plate['plateBoxes']
            plateFilename    = ("plate_%s.jpg" % str(plateIdx).zfill(7)).encode('utf-8')
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

            for box in plateBoxes:
                if int(self.occurrenceControl[str(box[4])]) > int(self.maxCharOccurrence) and self.balanceData == True:
                   continue

                xMin = box[0]
                yMin = box[1]
                xMax = box[2]
                yMax = box[3]

                xMins.append(float(xMin) / float(width))
                yMins.append(float(yMin) / float(height))
                xMaxs.append(float(xMax) / float(width))
                yMaxs.append(float(yMax) / float(height))
                classesText.append(str(box[4]).encode('utf-8'))
                classes.append(self.classes[str(box[4])])

                if not any(el['classID'] == self.classes[str(box[4])] for el in diffClasses):
                    diffClasses.append({"classID":self.classes[str(box[4])] , "className": str(box[4])})

                # increment occurrence counter
                self.occurrenceControl[str(box[4])] +=1

            # Avoid empty plates
            if len(classes) == 0:
                continue

            # Append data to TFRecord
            tfRecordExample                  = TFExample()
            tfRecordExample.width            = width
            tfRecordExample.height           = height
            tfRecordExample.filename         = plateFilename
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

        # sort label map and append to the pbtxt file
        diffClasses = sorted(diffClasses, key=lambda d: d['classID'], reverse=False)
        self.createTFLabelMap(diffClasses, tfLabelMapFilename)

        tfRecordGen.closeTfStream()
        elapsed = round((time() - startTime),3)
        print("TensorFlow dataset created successfully! Process took %s seconds" % (str(elapsed)))
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

    numOfPlates  = int(input("How many plates do you want to generate? \nNumber of plates:"))
    model        = int(input("0 - Tensorflow \n1 - YOLOV2\nWhat is the model? (e.g: 0 or 1): "))
    output       = input("What is the output path? ")
    augmentation = input("Want to augment the dataset? (y/n): ")
    balanced     = input("Want to balance the data? (You may have images with few annotations) (y/n): ")
    testSet      = input("Want to generate the test dataset as well? (y/n): ")
    showPlates   = input("Want to see generated plates? (y/n): ")

    if int(numOfPlates) > 0 and (model == 0 or model == 1) and output != "":
        if balanced == ('y' or 'Y'): balanced = True
        else: balanced = False

        if showPlates == ('y' or 'Y'): showPlates = True
        else: showPlates = False

        if augmentation == ('y' or 'Y'): augmentation = True
        else: augmentation = False

        if testSet == ('y' or 'Y'): testSet = True
        else: testSet = False

        # Create train set
        datasetCreator = DatasetCreator(numOfPlates, showPlates=showPlates, balanceData=balanced, augmentation=augmentation)

        if   model == 0:datasetCreator.createTensorFlowDataset(output)
        elif model == 1:datasetCreator.createYOLOV2Dataset()
        else: print("Model not found")

        if testSet:
            output = output + 'Test'
            datasetCreator = DatasetCreator(numOfPlates, showPlates=showPlates, balanceData=balanced, augmentation=augmentation, testSet=True)

            if model == 0:datasetCreator.createTensorFlowDataset(output)
            elif model == 1:datasetCreator.createYOLOV2Dataset()
            else:print("Model not found")

    else:
        print("Sorry, you chose something that does not match the requirements!")