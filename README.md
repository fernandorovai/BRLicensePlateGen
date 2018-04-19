# BR License Plate Generator
Generates artificial vehicle license plate following the Brazilian traffic department design patterns.

![plategendemo](https://user-images.githubusercontent.com/3229701/39006098-71177a7e-43d8-11e8-9533-a336a7a2e866.png)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
```
- Linux system: Ubuntu 16.04 (Xenial) or later
- python 3.5.2
- pip
- pillow
- matplotlib
- imgaug
- opencv
```

### Installing
Clone source code from git repo

```
$ git clone https://github.com/fernandorovai/BRLicensePlateGen
```

Install python dependencies via pip

```
$ pip install -r requirements.txt
```

## Running
```
$ cd BRLicensePlateGen
$ python plateGenerator.py 6 (generate 6 random plates)

- Augmentation is performed automatically in every generated plate.

----------------------------------------------------------
For external data usage, instantiate the class as follows:

from plateGenerator import plateGenerator
plateGen = PlateGenerator()
plates = plateGen.generatePlates(numOfPlates=numOfPlates)
- Plates dict structure
plates = {
          "plateIdx": idx,
          "plateImg": finalImg,
          "plateBoxes": [(xMin, yMin, xMax, yMax, tagValue)]
         }

plateIdx  = generated plate id
plateImg  = generated plate image

```

## Generating dataset file
Currently only tensorflow (TFRecord) export format is available.
- **Augmenting data:** rotating, scaling, adding gaussian blur, noise
- **Balancing data:** keep the number of characters balanced. It may result some images with a few bounding boxes.

```
$ cd BRLicensePlateGen
$ python datasetCreator.py

----------------------------------------------------------
Expected output:

How many plates do you want to generate?
Number of plates:5000
0 - Tensorflow
1 - YOLOV2
What is the model? (e.g: 0 or 1): 0
What is the output path? /home/user/BRLicensePlateGen/datasetFile
Want to augment the dataset? (y/n): y
Want to balance the data? (You may have images with few annotations) (y/n): y
Want to see generated plates? (y/n): n
------------------------------------------------------------------
Generating TensorFlow Dataset with (5000) license plates
TensorFlow dataset created successfully! Process took 19.082 seconds
```

## Built With

* [Pip](https://pip.pypa.io/en/stable/) - Dependency Management
* [Pillow](https://pillow.readthedocs.io/en/3.0.x/installation.html) - Image Handler

## Authors

* **Fernando Rodrigues Jr** - *Initial work* - [Fernando](https://github.com/fernandorovai)
