# ========================================================================= #
# Generates labels in the yolov5 format.                                    #
# Inputs: Annotation files in the COCO format                               #
#                                                                           #
# Allows for any image id datatype, not only integers.                      #
# ========================================================================= #

import numpy as np
import random
import json
import csv
from shutil import copyfile
import os
from collections import defaultdict

from pycocotools.coco import COCO


class Dataset:
    def __init__(self, filename):
        path = "../annotations/"
        self.filename = filename
    
        # Load annotations
        f = open(path + filename + ".json")
        anns = json.load(f)

        # Generate index
        # Two hash tables: 
        # Image_id as key, and annotation ids as values {list}
        # Annotation id as key and annotation objects as value
        self.img_ids_to_ann_ids = defaultdict(list)
        self.ann_id_to_anns = dict()
        self.img_ids_to_imgs = dict()

        self.img_ids = []
        for ann in anns['images']:
            self.img_ids.append(ann['id'])
            self.img_ids_to_imgs[ann['id']] = ann

        for ann in anns['annotations']:
            img_id = ann['image_id']
            ann_id = ann['id']

            self.img_ids_to_ann_ids[img_id].append(ann_id)
            self.ann_id_to_anns[ann_id] = ann
            
        assert(len(self.img_ids_to_ann_ids) == len(self.img_ids))       
        print("Loaded {} annotations from {} images!".format(len(self.ann_id_to_anns), len(self.img_ids)))      

    def get_annotations(self, img_id):
        """
        Gets a list of annotations for the given image_id.
        """
        ann_ids = self.img_ids_to_ann_ids[img_id]
        anns = []
        for ann_id in ann_ids:
            anns.append(self.ann_id_to_anns[ann_id])
        
        return anns

    def get_image_ids(self):
        return self.img_ids
    
    def get_image(self, img_id):
        return self.img_ids_to_imgs[img_id]


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
    # Traffic_light has been replaced by the three new categories traffic_light_red (92), traffic_light_green (93), and traffic_light_na (94)
    coco_to_yolo = {'1':'0', '2':'1', '3':'2', '4':'3', '6':'4', '7':'5', '8':'6', '11': '7', '13':'8', '17':'9', '18':'10', '92':'11', '93':'12', '94':'13'}
    #classes_yolo = {'0':'person', '1':'bicycle', '2':'car', '3':'motorcycle', '4':'bus', '5':'train', '6':'truck', 
    # '7':'fire hydrant', '8':'stop sign', '9':'cat', '10':'dog', '11':'traffic_light_red', 
    # '12':'traffic_light_green', '13':'traffic_light_na'}

    # Initialize COCO api for instance annotations
    filename = "instances_" + dataset_name
    data = Dataset(filename)
    img_ids = data.get_image_ids()


    # One file with filename = image_id containg all annotations
    for img_id in img_ids:
        if  "--" not in str(img_id):
            filename = (str(img_id)+'.txt').zfill(16) # Filenames have to be 12 characters long
        else:
            filename = (str(img_id)+'.txt')

        categoryId = []
        boxX = []
        boxY = []
        boxH = []
        boxW = []

        # Load annotations and images
        anns = data.get_annotations(img_id)
        if len(anns) == 0:
            break

        #img = coco.loadImgs(imgId)
        img = data.get_image(img_id)

        # For each annotation:
        for ann in anns:
            
            if str(ann['category_id']) != "10":
                categoryId.append(coco_to_yolo[str(ann['category_id'])])

                bbox_yolo = box_coco_to_yolo(ann['bbox'], img)
                boxX.append(bbox_yolo[0])
                boxY.append(bbox_yolo[1])
                boxH.append(bbox_yolo[2])
                boxW.append(bbox_yolo[3])

        assert(len(categoryId) == len(boxX) == len(boxY) == len(boxH) == len(boxW))
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
