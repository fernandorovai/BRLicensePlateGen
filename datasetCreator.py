#This script generates dataset based on a specific framework structure
import io
from plateGenerator import PlateGenerator
from TFRecordWriter import TFRecordWriter, TFExample
from time import time

class DatasetCreator:
    def __init__(self, numOfPlates, showPlates=False):
        plateGen = PlateGenerator(showPlates=showPlates)
        self.plates = plateGen.generatePlates(numOfPlates=numOfPlates)

    def createYOLOV2Dataset(self):
        # To be defined
        print("This feature is under development")

    def createTensorFlowDataset(self, tfRecordPath):
        startTime = time()
        print("------------------------------------------------------------------")
        print("Generating TensorFlow Dataset with (%d) license plates" % len(self.plates))
        tfRecordGen = TFRecordWriter(tfRecordPath)

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
                xMin = box[0]
                yMin = box[1]
                xMax = box[2]
                yMax = box[3]

                xMins.append(float(xMin) / float(width))
                yMins.append(float(yMin) / float(height))
                xMaxs.append(float(xMax) / float(width))
                yMaxs.append(float(yMax) / float(height))
                classesText.append(str(box[4]).encode('utf-8'))
                classes.append(ord(str(box[4])))

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

        tfRecordGen.closeTfStream()
        elapsed = round((time() - startTime),3)
        print("TensorFlow dataset created successfully! Process took %s seconds" % (str(elapsed)))

if __name__ == '__main__':

    numOfPlates = int(input("How many plates do you want to generate? \nNumber of plates:"))
    model       = int(input("0 - Tensorflow \n1 - YOLOV2\nWhat is the model? (e.g: 0 or 1): "))
    output      = input("What is the output path? ")
    showPlates  = input("Want to see generated plates? (y/n): ")

    if int(numOfPlates) > 0 and (model == 0 or model == 1) and output != "":
        if showPlates ==  ('y' or 'Y'):
            datasetCreator = DatasetCreator(numOfPlates, showPlates=True)
        else:
            datasetCreator = DatasetCreator(numOfPlates, showPlates=False)

        if   model == 0:
            datasetCreator.createTensorFlowDataset("%s.tfrecord" % output)
        elif model == 1:
            datasetCreator.createYOLOV2Dataset()
        else:
            print("Model not found")
    else:
        print("Sorry, you chose something that does not match the requirements!")