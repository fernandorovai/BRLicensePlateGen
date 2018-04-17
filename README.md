# BR License Plate Generator
Generates artificial vehicle license plate following the Brazilian traffic department design patterns

![plategenexample](https://user-images.githubusercontent.com/3229701/38843794-0d3db238-41c7-11e8-8fad-c70c19f73270.png)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
```
- Linux system: Ubuntu 16.04 (Xenial) or later
- python 3.5.2
- pip
- pillow
- matplotlib
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
## Built With

* [Pip](https://pip.pypa.io/en/stable/) - Dependency Management
* [Pillow](https://pillow.readthedocs.io/en/3.0.x/installation.html) - Image Handler

## Authors

* **Fernando Rodrigues Jr** - *Initial work* - [Fernando](https://github.com/fernandorovai)
