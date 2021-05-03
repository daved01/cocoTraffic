from pycocotools.coco import COCO

import numpy as np
import random
import json
import csv
from shutil import copyfile
import os

# Helper functions
def box_coco_to_yolo(bbox_coco, img):
    """
    Transforms a bounding box from COCO style to normalized YOLO style.
    COCO: x, y, h, w
    YOLO: x_center, y_center, width, height

    Args:
    bbox_coco   -- List of bounding box coordinates in coco format: [x,y,w,h] with x,y at the top centre.
    img         -- Dictionary with image meta data.

    Returns:
    bbox_yolo   -- List of bbox coordinates in yolo format.
    """


    bbox_yolo = []
    bbox_yolo.append((bbox_coco[0] + 0.5 * bbox_coco[2])/img['width'])
    bbox_yolo.append((bbox_coco[1] + 0.5 * bbox_coco[3])/img['height'])
    bbox_yolo.append(bbox_coco[2]/img['width'])
    bbox_yolo.append(bbox_coco[3]/img['height'])

    return bbox_yolo


def run(dataset_name):
    # Set category_id mapping
    # To accomodate our 15 classes, the category_ids are remapped before writing them to the labels for yolo.
    # TODO: Delete traffic_light (10) since this will be refined into 92 etc.
    #classes = {'1': 'person','2': 'bicycle', '3': 'car', '4': 'motorcycle', '6': 'bus', '7': 'train', '8': 'truck', '10': 'traffic light',
    #'11': 'fire hydrant', '13': 'stop sign', '17': 'cat', '18': 'dog', '92':'traffic_light_red', '93':'traffic_light_green', '94': 'traffic_light_na'}

    # Traffic_light has been replaced by the three new categories traffic_light_green (92), traffic_light_red (93), and traffic_light_na (94)
    coco_to_yolo = {'1':'0', '2':'1', '3':'2', '4':'3', '6':'4', '7':'5', '8':'6', '11': '7', '13':'8', '17':'9', '18':'10', '92':'11', '93':'12', '94':'13'} #TODO: Remove coco 10 once the new labels are available

    #classes_yolo = {'1':'person', '2':'bicycle', '3':'car', '4':'motorcycle', '5':'bus', '6':'train', '7':'truck', '8':'fire hydrant', '9':'stop sign', '10':'cat', '11':'dog', '12':'traffic_light_red', '13':'traffic_light_green', '14':'traffic_light_na', }

    # Load coco dataset
    dataDir='..'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset_name)

    # Initialize COCO api for instance annotations
    coco=COCO(annFile)

    # Load COCO categories and supercategories
    imgIds = coco.getImgIds()

    print('Number of images: ' + str(len(imgIds)))

    # Read annotation file
    # One file with filename = image_id containg all annotations
    for imgId in imgIds:
        filename = (str(imgId)+'.txt').zfill(16) # Filenames have to be 12 characters long
        categoryId = []
        boxX = []
        boxY = []
        boxH = []
        boxW = []

        annIds = coco.getAnnIds(imgId)
        anns = coco.loadAnns(annIds)
        img = coco.loadImgs(imgId)

        # For each annotation:
        for ann in anns:
            
            if str(ann['category_id']) != "10":
                categoryId.append(coco_to_yolo[str(ann['category_id'])])

                bbox_yolo = box_coco_to_yolo(ann['bbox'], img[0])
                boxX.append(bbox_yolo[0])
                boxY.append(bbox_yolo[1])
                boxH.append(bbox_yolo[2])
                boxW.append(bbox_yolo[3])

        # Write file
        file_path = '../labels/' + dataset_name + '/'
        
        rows = zip(categoryId, boxX, boxY, boxH, boxW)
        
        with open(file_path+filename, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            for row in rows:
                writer.writerow(row)
 

if __name__=="__main__":
    dataset_name = 'trainTraffic'
    run(dataset_name)