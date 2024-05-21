# Data-centrIc semAntic segMentation to mAp iNfestations in saTellite imagEs  (DIAMANTE)


The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Dino Ienco, Vito Recchia_

[DIAMANTE: A data-centric semantic segmentation approach to map tree dieback induced by bark beetle infestations via satellite images]() 

Please cite our work if you find it useful for your research and work.
```
 @article{,
 title = {DIAMANTE: A data-centric semantic segmentation approach to map tree dieback induced by bark beetle infestations via satellite images},
journal = {},
volume = {},
pages = {},
year = {},
issn = {},
doi = {},
url = {},
author = {}}

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



To launch the dataset pipeline download on your own you have to launch:
```
python Data/planetary_download.py
```
You have also to specify these parameters into ```planetary_download.py``` **main()** function:
* **base_out_path**: defines the output path, where the Sentinel 1 and Sentinel 2 images will be downlaoded 
* **base_geojson_path**: the path where the geojson defining each scene is stored 
* **start_date** and **end_date**: defining the search interval for Sentinel 2 search images.

All the required packages needed to run the download pipeline are described in ```Data/requirements.txt``` file.


## How to use

The repository contains the following scripts:
* main.py:  script to execute RDIAMANTE


## Replicate the experiments
Modify the following code in the main.py script to change the beaviour of ROULETTE
Link to the learned models [MODELS]((https://drive.google.com/drive/folders/1j0nxUXlBo2dUZin5sXupeGxMx2N1-slh?usp=sharing))










