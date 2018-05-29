"""THIS FILE IS RESPONSIBLE FOR READING A TFRECORD FILE
AND REGENERATING DATA
"""

import io
import tensorflow as tf
import os
from PIL import Image
from MkDataSetStructure import MkDataSetStructure
from Tagger import Tagger

# Reading an existent tfrecord and extract information
class TFRecordReader:
    def __init__(self, tfrecordsFilename):
        self._readerIterator  = tf.python_io.tf_record_iterator(path=tfrecordsFilename)

    def readTFRecord(self):
        dataBuffer = []
        for stringRecord in self._readerIterator:
            tempData = {}
            example = tf.train.Example()
            example.ParseFromString(stringRecord)

            height          = int(example.features.feature['image/height'].int64_list.value[0])
            width           = int(example.features.feature['image/width'].int64_list.value[0])
            filename        = example.features.feature['image/filename'].bytes_list.value[0]
            sourceID        = example.features.feature['image/source_id'].bytes_list.value[0]
            imgEncoded      = example.features.feature['image/encoded'].bytes_list.value[0]
            imageFormat     = example.features.feature['image/format'].bytes_list.value[0]
            xMins           = example.features.feature['image/object/bbox/xmin'].float_list.value
            xMaxs           = example.features.feature['image/object/bbox/xmax'].float_list.value
            yMins           = example.features.feature['image/object/bbox/ymin'].float_list.value
            yMaxs           = example.features.feature['image/object/bbox/ymax'].float_list.value
            classesText     = example.features.feature['image/object/class/text'].bytes_list.value
            classesID       = example.features.feature['image/object/class/label'].int64_list.value

            tempData["height"]      = height
            tempData["width"]       = width
            tempData["filename"]    = filename.decode("utf-8")
            tempData["sourceID"]    = sourceID.decode("utf-8")
            tempData["imgEncoded"]  = imgEncoded
            tempData["imageFormat"] = imageFormat
            tempData["xMins"]       = [(i * width) for i in xMins]
            tempData["xMaxs"]       = [(i * width) for i in xMaxs]
            tempData["yMins"]       = [(i * height) for i in yMins]
            tempData["yMaxs"]       = [(i * height) for i in yMaxs]
            tempData["classesText"] = classesText
            tempData["classesID"]   = classesID

            dataBuffer.append(tempData)
        return dataBuffer

    # regenerate original image file from tfrecord data
    def regenerateImages(self, outputPath):
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)

        dataBuffer = self.readTFRecord()
        for data in dataBuffer:
            rawImageData            = data["imgEncoded"]
            height                  = data["height"]
            width                   = data["width"]
            filename                = data["filename"]

            self.saveFromRawImageData(rawImageData, height, width, outputPath, filename)

    def saveFromRawImageData(self, raw1DImageData, height, width, outputPath, filename):
        img = Image.open(io.BytesIO(raw1DImageData))
        img.save(os.path.join(outputPath, filename))

    def tfRecordToCaffe(self, datasetName, outputPath, nameAsGroundTruth=False):
        MkDataSetStructure(os.path.join(outputPath,datasetName))
        fileManager = Tagger(os.path.join(outputPath, datasetName))

        dataBuffer = self.readTFRecord()
        for data in dataBuffer:
            if nameAsGroundTruth: imageName = data['filename']
            else: imageName = data["sourceID"]

            imageFilename = imageName + ".jpg"
            imageOutputPath = os.path.join(outputPath, datasetName, "Images")

            fileManager.AppendTrainingImg(imageName)
            self.saveFromRawImageData(data["imgEncoded"],
                                      data["height"],
                                      data["width"],
                                      imageOutputPath,
                                      imageFilename)

            for xMin, yMin, xMax, yMax, classText, classID in zip(data["xMins"], data["yMins"],
                                                                  data["xMaxs"], data["yMaxs"],
                                                                  data["classesText"], data["classesID"]):
                fileManager.AppendAnnotation((xMin, yMin), (xMax,yMax), imageName, classText.decode('utf-8') + " " + str(classID))

if __name__ == "__main__":
    outputPath = '/home/junior/Documents/NN/datasets'
    tfReader = TFRecordReader('/home/junior/Documents/Personal/BRLicensePlateGen/licensePlate7kAug.tfrecord')
    # tfReader.regenerateImages(outputPath)
    tfReader.tfRecordToCaffe("licensePlate7kAug", outputPath, nameAsGroundTruth=True)