# Data-centrIc semAntic segMentation to mAp iNfestations in saTellite imagEs  (DIAMANTE)


The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Dino Ienco, Vito Recchia_

[DIAMANTE: A data-centric semantic segmentation approach to map tree dieback induced by bark beetle infestations via satellite images]() 

Please cite our work if you find it useful for your research and work.
```
 @article{,
 title = {DIAMANTE: A data-centric semantic segmentation approach to map tree dieback induced by bark beetle infestations via satellite images},
journal = {Journal of Intelligent Information Systems},
volume = {},
pages = {},
year = {2024},
issn = {},
doi = {https://doi.org/10.1007/s10844-024-00877-6},
url = {https://link.springer.com/article/10.1007/s10844-024-00877-6},
author = {G.Andresini, A. Appice, D.Ienco, V.Recchia}}

```


## Code requirements
The code relies on the following python3.7+ libs.
Packages needed are:
* Tensorflow 2.4.0
* Pandas 1.3.2
* Numpy 1.19.5
* Matplotlib 3.4.2
* Hyperopt 0.2.5
* Keras 2.4.3
* Scikit-learn 0.24.2



## Data
The following [DATASETS]([https://drive.google.com/drive/folders/1NbkApPwjgzWX2s-zZEb7gSDP1oT3Ea4m?usp=sharing](https://drive.google.com/drive/folders/16u83KnN996dollCMzUdgOEL6do6p5NJr?usp=drive_link)).
The datasets used are:

## Models
The models used for the experimental setting are reported in [MODELS](https://drive.google.com/drive/folders/1j0nxUXlBo2dUZin5sXupeGxMx2N1-slh?usp=sharing)

## How to use the script to download dataset

To launch the dataset pipeline download on your own you have to launch:
```
python Data/planetary_download.py
```
You have also to specify these parameters into ```planetary_download.py``` **main()** function:
* **base_out_path**: defines the output path, where the Sentinel 1 and Sentinel 2 images will be downlaoded 
* **base_geojson_path**: the path where the geojson defining each scene is stored 
* **start_date** and **end_date**: defining the search interval for Sentinel 2 search images.

All the required packages needed to run the download pipeline are described in ```Data/requirements.txt``` file.


## How to use the code to run model

The repository contains the following scripts under the folder Model:
* main.py:  script to execute DIAMANTE

* Please specific the parameter of the dataset as reported in __CONFIG.conf__  file . E.g.,  To run the code the command is main.py NameOfDataset (es Sentinel12)


## Replicate the experiments


To replicate experiments reported in the work, you can use models and datasets stored in homonym folders.
Global variables are stored in __CONFIG.conf__  file 


```python
[Sentinel2]
pathModels = ../Models/Sentinel_2/
pathDatasetTrain = ../DS/SWIFT/planetary/sentinel_2/Train/
pathDatasetTest = ../Datasets/Sentinel_2/
pathDatasetTrainM = ../DS/SWIFT/planetary/Masks/Train/
pathDatasetTestM = ../Datasets/Masks/
nameModel=unet_resize_1_model.h5
#this is to remove channel 13 (SCL) to the tiff files
sizetest=266
resizeChannel=1
pathModelPretrained=Models/
shape=32
channels=12
channels1=0
tilesSize=32
attack=1
tiles=1




[setting]
ATTACK_LABEL = 1
PREPROCESSING=1
PREPROCESSING_MASKS=1
# if 0 the U-Net is trained
LOAD_NN = -1
#1 to perform experiment with middle and late configurations
TRAIN_LATEUNET= 0
RESIZE=1
#1 to perform prediction with trained model specified in the pathModels
PREDICTION=1
#if 1 perform prediction with middle and late unet otherwise it is performed with early stage
PREDICT_LATEUNET=0
#if 1 the late U-net is used otherwise hybrid for train and prediction
LATE=0
# if 1 the training and prediciton is performed with SUM operator fusion otherwise it is done with CONC operator
SUM=1


```









