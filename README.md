# COCO Dataset Extensions

We refined the traffic light class (index 10) of the COCO dataset into the three classes traffic_ligt_red (92), traffic_light_green (93), traffic_light_na (94).

We integrated these relabelled data into three datasets.

We provide two ways to access the refined labels. You can use the `setup.py` to automatically download the annotation files, or you can do it manually following the steps below.

In all cases you have to download all images train2017 (20GB) from the COCO [website]() first and put the folder into the folder `images`.

To learn more about the dataset see our post [here]().


## 1. COCO Traffic Lights
Contains all COCO images which contain traffic lights, with all of them refined.

+ PLOT


4330 images
{10: 927, 92: 3302, 93: 1877, 94: 7415}


Get the annotation files [here]().

## 2. COCO Traffic Full
Full COCO 2017 dataset, with all traffic lights relabelled in training and validation dataset. You can access the annotations [here](link to drive .zip of both files)
Once you have downloaded them put them into the folder `annotations`.


## 3. COCO Traffic Small
Subset of the train2017 images with classes which are related to traffic, and all traffic light images from val2017.

+ PLOT

Ideal to train a smaller model to detect vehicles, pedestrians etc.


## 4. COCO Traffic Plus
Like 3, with images of traffic lights added from the LISA Traffic Light images labelled and added.

+ PLOT

