# Create the dataset structure
# dataset/Images
# dataset/Annotations
# dataset/TestOutput
# dataset/ImageSets

import os

class MkDataSetStructure():
    def __init__(self, path):
        self.path = path
        self.CreateFolders()

    def CreateFolders(self):
        #Dataset
        if not os.path.isdir(self.path):
           os.mkdir(self.path)
        else:
            print("Dataset Already Exists!")
            return

        #Images
        if not os.path.isdir(os.path.join(self.path, "Images")):
           os.mkdir(os.path.join(self.path, "Images"))

       #Annotations
        if not os.path.isdir(os.path.join(self.path, "Annotations")):
           os.mkdir(os.path.join(self.path, "Annotations"))

       #ImageSets
        if not os.path.isdir(os.path.join(self.path, "ImageSets")):
           os.mkdir(os.path.join(self.path, "ImageSets"))

       #Images
        if not os.path.isdir(os.path.join(self.path, "TestOutput")):
           os.mkdir(os.path.join(self.path, "TestOutput"))

        # ImageLogs
        if not os.path.isdir(os.path.join(self.path, "ImageLogs")):
            os.mkdir(os.path.join(self.path, "ImageLogs"))

        print("Dataset Structure Created Successfully! %s" % str(self.path))
if __name__ == "__main__":
    MkDataSetStructure(raw_input("What is the dataset name? "))