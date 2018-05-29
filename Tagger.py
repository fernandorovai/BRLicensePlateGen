# CLASS RESPONSIBLE FOR HANDLING DATA AND FILES

import os
from os import listdir
from os.path import isfile

class Tagger:
    def __init__(self, dataSet_dir):

        #Setting directories
        self._tagDir = os.path.join(dataSet_dir, "Annotations")
        self._outputDir = os.path.join(dataSet_dir, "TestOutput")
        self._imageSetsDir = os.path.join(dataSet_dir, "ImageSets")
        self._imageLogDir = os.path.join(dataSet_dir, "ImageLogs")
        self._imagesDir = os.path.join(dataSet_dir, "Images")

    def AppendAnnotation(self, leftBottom, rightTop, imgName, cls):
        tagFile = os.path.join(self._tagDir, '%s.txt' % (imgName))
        logData = open(tagFile, 'a+')

        xMin = leftBottom[0]
        xMax = rightTop[0]
        yMin = leftBottom[1]
        yMax = rightTop[1]

        # Guarantee the convention leftBottom - rightTop.
        if int(leftBottom[0]) > int(rightTop[0]):
            xMin = rightTop[0]
            xMax = leftBottom[0]

        if int(leftBottom[1]) > int(rightTop[1]):
            yMin = rightTop[1]
            yMax = leftBottom[1]

        # CHECK ZERO AND NEGATIVE COORD
        if int(leftBottom[0]) <= 0 or int(leftBottom[1]) <= 0 or int(rightTop[0]) <= 0 or int(rightTop[1]) <= 0:
            print("Problem neg coord found here: " + str(imgName) + "file: " + "\n")



        logData.write(str(int(xMin)) + " " + str(int(yMin)) + " " + str(int(xMax)) + " " + str(int(yMax)) + " " + str(cls) + "\n")
        logData.close()

    def AppendTrainingImg(self, imgName):
        trainFile = os.path.join(self._imageSetsDir, "train.txt")
        stringExists = self.CheckExistence(trainFile, imgName)

        if not stringExists:
            trainData = open(trainFile, 'a+')
            trainData.write(str(imgName) + "\n")
            trainData.close()

            # trainData = open(trainFile, 'a+')
            # # sort according to filename
            # data = trainData.readlines()
            # data.sort()
            #
            # self.EraseTrainingFile()
            # for img in data:
            #     trainData.write(str(img))
            # trainData.close()

    def AppendClassName(self, className):
        classesFile = os.path.join(self._imageSetsDir, "classes.txt")
        stringExists = self.CheckExistence(classesFile, className)

        if not stringExists:
            classesData = open(classesFile, 'a+')
            classesData.write(str(className))
            classesData.close()
            classesData.close()

    def AppendImgLog(self, imgName, log):
        logFile = os.path.join(self._imageLogDir, '%s.txt' % (imgName))
        logData = open(logFile, 'w+')
        logData.write(str(log))
        logData.close()

    def AppendTestImg(self, imgName):
        fileName = os.path.join(self._imageSetsDir, "test.txt")
        stringExists = self.CheckExistence(testFile, imgName)

        if not stringExists:
            testData = open(testFile, 'a+')

            testData.write(str(imgName))
            testData.close()

            testData = open(testFile, 'a+')
            #sort according to filename
            data = testData.readlines()
            data.sort()
            self.EraseTestFile()
            for img in data:
                testData.write(str(img))
            testData.close()

    def LoadAnnotationsData(self, imgName):
        fileName = os.path.join(self._tagDir, '%s.txt' % (imgName))
        if os.path.exists(fileName):
            tagData = open(fileName, 'r').readlines()
            return tagData
        return False

    def LoadTrainingData(self):
        fileName = self._imageSetsDir + "train.txt"
        if os.path.exists(fileName):
            trainingData = open(fileName, 'r').readlines()
            return trainingData
        return False

    def LoadImgLog(self, imgName):
        fileName = os.path.join(self._imageLogDir, '%s.txt' % (imgName))
        if os.path.exists(fileName):
            tagData = open(fileName, 'r').readlines()
            return tagData
        return False

    def LoadTestData(self):
        fileName = self._imageSetsDir + "test.txt"
        if os.path.exists(fileName):
            testData = open(fileName, 'r').readlines()
            return testData
        return False

    def LoadOutputData(self, imgName):
        fileName = os.path.join(self._outputDir, '%s.txt' % (imgName))
        if os.path.exists(fileName):
            tagData = open(fileName, 'r').readlines()
            return tagData
        return False

    def LoadDataSetImages(self):
        imageDataSet = []
        dirList = listdir(self._imagesDir)

        for file in dirList:
            if isfile(os.path.join(self._imagesDir, file)):
                fileType = os.path.splitext(file)[1]
                if fileType == '.jpg' or fileType == '.JPG':
                    # rawName = os.path.splitext(file)[0]
                    imageDataSet.append(file)
        return imageDataSet

    def LoadClasses(self):
        fileName = os.path.join(self._imageSetsDir, "classes.txt")
        if os.path.exists(fileName):
            classes = open(fileName, 'r').readlines()
            return classes
        return False

    def EraseClassesFile(self):
        fileName = os.path.join(self._imageSetsDir, "classes.txt")
        if os.path.exists(fileName):
            open(fileName, 'w').close()

    def EraseAnnotations(self, imgName):
        fileName = os.path.join(self._tagDir, '%s.txt' % (imgName))
        if os.path.exists(fileName):
            open(fileName, 'w').close()

    def EraseTrainingFile(self):
        fileName = os.path.join(self._imageSetsDir, "train.txt")
        if os.path.exists(fileName):
            open(fileName, 'w').close()

    def EraseTestFile(self):
        fileName = self._imageSetsDir + "test.txt"
        if os.path.exists(fileName):
            open(fileName, 'w').close()

    @staticmethod
    def CheckExistence(filename, string):
        if not os.path.isfile(filename):
            return False

        if string in open(filename).read():
            return True
        return False
