# COCO Dataset Extensions

We refined the traffic light class (index 10) of the COCO dataset into the three classes, traffic_light_red (92), traffic_light_green (93), traffic_light_na (94), and integrated these into three datasets (see below). We discovered mislabelled traffic lights in the original COCO 2017 training and validation data and kept their label as 10, so that these annotations can easily be identified.

To learn more about the dataset see our post [here]().

If you want to gnerate annotations for [yolov5](https://github.com/ultralytics/yolov5) from COCO annotation files, use the file `make_yolo_labels.py`.

## Setup
The setup varies for each of the three datasets since they require different files. For each follow the steps desribed below.

In all cases, create folders `annotations` and `images` at the root of this repository first. Then, download the images `train2017` and `val2017` from the COCO [website]() and add them to the `images` folder. Then your repository should look like this:

```
cocoTraffic/
├── annotations/
├── api/
│   ├── make_datasets.py
│   ├── make_yolo_labels.py
│   ├── ...
├── images/
│   ├── train2017/
│       ├── 000000000009.jpg
│       ├── ...
├── plots/
│   ├── ...
├── tools/
│   ├── ...
├── LICENSE
├── README.md
```

With this base setup choose the dataset that you need and follow the instructions.


## 1. COCO Refined
Full COCO 2017 dataset, with all traffic lights relabelled in training and validation dataset. Get the annotation files with the refined labels [here](https://drive.google.com/file/d/1weZpzmva_fcTtiSIm9jdM73PdBoJgzOe/view?usp=sharing) and place them into the `annotations` folder. 


## 2. COCO Traffic
Subset of the `train2017` images with classes which are related to traffic, and all traffic light images from `val2017`. These images have been split into a training and validation set (80/20). The chosen classes are:

['traffic light', 'car', 'truck','bus', 'motorcycle', 'bicycle', 'person', 'dog', 'cat', 'stop sign', 'fire hydrant', 'train', 'traffic_light_red', 'traffic_light_green', 'traffic_light_na']

Ideal to train a smaller model to detect vehicles, pedestrians etc. Annotation files are availabe [here](https://drive.google.com/file/d/1Oust5GrOrzP7588_ZS5Qb6cgWf5FhSN3/view?usp=sharing).


## 3. COCO Traffic Extended
Extended COCO Traffic with images of traffic lights from the [LISA Traffic Light](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset) dataset. We labelled these images to include all classes from `COCO Traffic`. Annotation [file](https://drive.google.com/file/d/1ibviz00vjHelwkkoJfQEhTmtrrw9p7Wx/view?usp=sharing)



# Tools
To label the data, we created and used the following tools.

`dataLabeller` - Tool which iterates through COCO annotations and lets you change their category id. Used to relabel the traffic lights.

`makesense` - Makesense is a freely [available](https://www.makesense.ai) annotation tool which we used to label the images in the LISA Traffic Lights dataset. We include a file which converts the output from `makesense.ai` into a COCO dataset annotation file.

`prelabeller` - DETR model to label data with COCO classes. We used it to prelabel the LISA Traffic Light images.