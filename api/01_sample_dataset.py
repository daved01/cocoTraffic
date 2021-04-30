import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import random
import json
import csv

from pycocotools.coco import COCO


#%% Load train2017
def sample_train2017():
    dataDir='..'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    # Initialize COCO api for instance annotations
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())

    # Map category ids to names
    catIds = coco.getCatIds()
    catNames = dict()
    for i in range(len(catIds)):
        catNames[str(cats[i]['id'])] = cats[i]['name']

    assert(len(catIds) == 80) # Should be 80 classes
    assert(len(catNames) == 80) # Should be 80 classes

    ## Sample images and annotations
    imgIds_traffic = []
    anns_traffic = []


    # Sample traffic light images
    print("-----------\nSampling traffic light images...")
    catIds = coco.getCatIds(catNms='traffic light')
    assert(catIds == [10])
    imgIds_all_traffic_lights = coco.getImgIds(catIds=catIds)
    print('Total number of images: {}'.format(len(imgIds_all_traffic_lights)))

    # Sample 1000 images from all images containing traffic lights
    num_samples = 1000
    random.seed(2004)
    imgIds_sampled_traffic_lights = random.sample(imgIds_all_traffic_lights, num_samples)
    assert(len(imgIds_sampled_traffic_lights) == 1000)
    # Add them to the final list of images
    imgIds_traffic = imgIds_sampled_traffic_lights
    print('Sampled {} images.'.format(num_samples))
    # Load annotations for the sampled images
    annIds_sampled_traffic_lights = coco.getAnnIds(imgIds_sampled_traffic_lights)
    anns_sampled_traffic_lights = coco.loadAnns(annIds_sampled_traffic_lights) # List of dictionaries for each annotation
    print('Number of object instances in the {} images: {}'.format(num_samples, len(annIds_sampled_traffic_lights)))
    # Add annotations to final list of annotations
    anns_traffic = anns_sampled_traffic_lights

    # Sample other classes
    num_samples_should = num_samples
    classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'dog', 'cat', 'stop sign', 'fire hydrant', 'train']
    for cl in classes:
        num_samples_should += num_samples
        print("-----------\nSampling {}...".format(cl))
        catIds = coco.getCatIds(catNms=cl)
        imgIds = coco.getImgIds(catIds=catIds)
        print('Total number of images containing object {}: {}'.format(cl, len(imgIds)))

        # Sample num_samples number of images from all images with specified class
        imgIds_sampled = random.sample(imgIds, num_samples)
        assert(len(imgIds_sampled) == 1000)

        # Add them to the final list of images IDs and remove redundant images
        for img in imgIds_sampled:
            imgIds_traffic.append(img)  
        imgIds_traffic = list(set(imgIds_traffic))
        
        # Fill up rest with images
        num_resample = num_samples_should - len(imgIds_traffic)
        while num_resample > 0:
            sample = random.sample(imgIds, 1)[0]
            if sample not in imgIds_traffic:
                imgIds_traffic.append(sample)
                num_resample = num_samples_should - len(set(imgIds_traffic))

        assert(len(imgIds_traffic) == num_samples_should)
        print('Sampled {} images containing object {}.'.format(num_samples, cl))

        # Load annotations for the sampled images
        #annIds_sampled = coco.getAnnIds(imgIds_sampled)
        #anns_sampled = coco.loadAnns(annIds_sampled) # List of dictionaries for each annotation
        #for ann in anns_sampled:
        #    anns_traffic.append(ann)
        #print('Number of object instances in the {} images: {}'.format(num_samples, len(annIds_sampled)))

    assert(len(imgIds_traffic) == 12000) # Should be 12000 images

    # Get annotations
    annIds_sampled = coco.getAnnIds(imgIds_traffic)
    anns_traffic = coco.loadAnns(annIds_sampled) # List of dictionaries for each annotation

    # Remove redundent annotations
    print("\nRemoving redundant annotations...")
    num_anns_bef = len(anns_traffic)
    anns_traffic = list({ann['id']:ann for ann in anns_traffic}.values())
    print("Removed {} redundant annotations from the dataset!".format(num_anns_bef - len(anns_traffic)))

    # Remove other classes from data
    print("\nRemoving out of scope classes...")
    class_names = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'dog', 'cat', 'stop sign', 'fire hydrant', 'train']
    anns_traffic_cleaned = []

    for ann in anns_traffic:
        if ann['category_id'] in coco.getCatIds(catNms=class_names):
            anns_traffic_cleaned.append(ann)
            
    print('Removed {} annototations from the data.\nNumber of images: {}\nNumber of annotations: {}'.format(len(anns_traffic)-len(anns_traffic_cleaned), 
    len(imgIds_traffic), len(anns_traffic_cleaned)))
    
    # Save annotations
    filename = "traffic"
    print("\nSaving annotations...")
     # Load dataset file to get structure
    target_file = json.load(open(annFile, 'r'))
    imgs = coco.loadImgs(imgIds_traffic)
    assert(len(imgs) == 12000)

    # Make final dictionary
    dataset = dict.fromkeys(target_file.keys())
    dataset['info'] = target_file['info']
    dataset['licenses'] = target_file['licenses']
    dataset['categories'] = target_file['categories']
    dataset['annotations'] = anns_traffic_cleaned
    dataset['images'] = imgs

    # Save to disk
    with open('../annotations/instances_'+str(filename)+'.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename))


def test_annotations():
    ann_file = "../annotations/instances_traffic.json"
    
    # Get imgs and anns
    f = open(ann_file)
    ann_file = json.load(f)
    
    img_ids = []
    for i in range(len(ann_file['images'])):
        img_ids.append(ann_file['images'][i]['id'])
    assert(len(img_ids) == 12000)
    

    anns = ann_file['annotations']
    anns_img_ids = []
    for ann in anns:
        anns_img_ids.append(ann["image_id"])

    anns_img_ids = set(anns_img_ids)
    img_ids = set(img_ids)

    assert(len(anns_img_ids) == 12000)
    assert(anns_img_ids == img_ids)
    print("Annotations ok")


if __name__ == "__main__":
    sample_train2017()
    test_annotations()