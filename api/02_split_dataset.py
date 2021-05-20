# ========================================================================= #
# Splits the previously sampled dataset into a train and val set.           #
# Input: Annotation file in COCO format                                     #
#                                                                           #
# ========================================================================= #

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import csv
from shutil import copyfile

from pycocotools.coco import COCO


def get_diff(l1, l2):
    """
    Returns the difference between two lists.
    """
    diff = list( list(set(l1) - set(l2)) + list(set(l2) - set(l1)) )

    return diff


def split_dataset(train=0.8):
    """
    Splits traffic dataset into the datasets train and val.
    """

    # Load dataset traffic
    ann_file = "../annotations/instances_traffic.json"
    f = open(ann_file)
    anns = json.load(f)

    img_ids = []
    for i in range(len(anns['images'])):
        img_ids.append(anns['images'][i]['id'])
    
    # Check length and for uniqueness
    assert(len(set(img_ids)) == 12000)

    # Check the annotations for classes
    num_train = int(len(img_ids) * train)
    num_val = len(img_ids) - num_train

    img_ids_train = random.sample(img_ids, num_train)
    img_ids_val = get_diff(img_ids, img_ids_train)

    assert(len(img_ids_train) == num_train)
    assert(len(img_ids_val) == num_val)
    
    return img_ids_train, img_ids_val


def get_annotations(img_ids_train, img_ids_val):
    """"
    Takes two lists of img ids and a list of annotations and saves
    corresponding annotations.
    """

    ann_file = "../annotations/instances_traffic.json"
    coco=COCO(ann_file)

    # Load annotations
    annIds_train = coco.getAnnIds(img_ids_train)
    anns_train = coco.loadAnns(annIds_train)

    annIds_val = coco.getAnnIds(img_ids_val)
    anns_val = coco.loadAnns(annIds_val)

    assert((len(anns_train) + len(anns_val)) == 80941)

    return anns_train, anns_val


def save_dataset(img_ids, anns, filename):
    # Load dataset val to get structure
    ann_file = "../annotations/instances_traffic.json"
    target_file = json.load(open(ann_file, 'r'))
    coco=COCO(ann_file)
    imgs = coco.loadImgs(img_ids)

    # Make final dictionary
    dataset = dict.fromkeys(target_file.keys())
    dataset['info'] = target_file['info']
    dataset['licenses'] = target_file['licenses']
    dataset['categories'] = target_file['categories']
    dataset['annotations'] = anns
    dataset['images'] = imgs

    # Save to disk
    with open('../annotations/instances_'+str(filename)+'.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename))


def copy_image_files(img_ids, foldername):
    path = '../images/'
    count = 0

    for img in img_ids:
        filename = (str(img) + ".jpg").zfill(16)

        try:
            copyfile(path+'train2017/'+filename, path+foldername+'/'+filename)
            count +=1
        except FileNotFoundError:
            print('Folder {} does not exist. Please create it and try again.'.format(path+foldername))
            return
        
    print('Copied {} images to {}'.format(count, path+foldername))
    

#%% TESTS:
def split_dataset_test():
    img_ids_train, img_ids_val = split_dataset()
    img_ids = img_ids_train + img_ids_val 
    assert(len(img_ids_train) == 9600)
    assert(len(img_ids_val) == 2400)
    assert(len(set(img_ids)) == 12000)

    

#%%
if __name__=="__main__":
    img_ids_train, img_ids_val = split_dataset()
    anns_train, anns_val = get_annotations(img_ids_train, img_ids_val)
    
    #save_dataset(img_ids_train, anns_train, "trainTraffic")
    #save_dataset(img_ids_val, anns_val, "valTraffic")

    #copy_image_files(img_ids_train, "trainTraffic")
    #copy_image_files(img_ids_val, "valTraffic")
    