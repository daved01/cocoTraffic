# =================================================================== #
# Appends the LISA annotations to the coco train and val data.        #
#                                                                     #
# LISA annotations are in the makesense.ai .csv format.               #
# =================================================================== #

import pandas as pd
import os
from shutil import copyfile
import random
import json


def get_diff(l1, l2):
    """
    Returns the difference between two lists.
    """
    diff = list( list(set(l1) - set(l2)) + list(set(l2) - set(l1)) )

    return diff


def load_LISA_annotations(makesense_annotation_files):
    """
    Loads all makesense.ai .csv files into a single pandas dataframe.
    """
    makesense_path = "./relabelled/"
    cols = ["label", "x", "y", "w", "h", "name", "size_w", "size_h"]
    makesense_anns = pd.DataFrame(columns=cols)

    for file in makesense_annotation_files:
        anns_part = pd.read_csv(makesense_path+file, names=cols)
        makesense_anns = makesense_anns.append(anns_part)

    makesense_anns.reset_index(inplace=True)

    return makesense_anns


def filter_lisa_anns(anns):
    """
    Filters out images which contain the not-relabelled coco traffic light
    class from the given dataframe. Returns a list of images which 
    contain the relabelled classes.
    """ 
    print("Found {} images.".format(len(set(anns['name']))))
    row_del = anns[anns['label'] == "traffic light"]
    imgs_del = list(set(row_del['name']))
    image_names = get_diff(list(set(anns['name'])), imgs_del)
    print("Removed {} images with unannotated traffic lights."
            .format(len(imgs_del)))

    return image_names
    
 
def copy_images_from_lisa(img_filenames, path_source):
    """
    Copies the images in the given list from the dataset folders
    as downloaded into a single  folder.
    """
    count = 0
    for filename in img_filenames:
    
        if filename.find("Clip") != -1:
            folder_name = filename.split('--')[0].replace("Clip", "Train")
            folder_name = "".join(filter(lambda x: not x.isdigit(), folder_name))
            # Format: nightTrain/nightTrain/nightClip2/frames/nightClip2--...
            path = path_source + folder_name + "/" + folder_name + "/" + filename.split('--')[0] +"/frames/"
        else:
            folder_name = filename.split('--')[0]
            path = path_source + folder_name + "/" + folder_name + "/frames/"
       
        try:
            copyfile(path+filename, "../images/TrafficLISA/"+filename)
            count +=1
        except FileNotFoundError:
            print('Folder {} does not exist. Please create it and try again.'.format(path))
            return
    assert(count == len(img_filenames))

    print('Copied {} images to {}'.format(count, path+"relabel"))


def split_anns(df_anns, split=0.8, copy_files=False):
    """
    Splits the data into train and val. Data is given as a dataframe.
    Copies the image files into folders if copy_file=True.
    """
    
    # Get all images and sample them
    img_files = list(set(df_anns['name']))  
    num_train = int(len(img_files) * split)
    num_val = len(img_files) - num_train
    
    # Sample
    random.seed(1867)
    imgs_train = random.sample(img_files, num_train)
    imgs_val = get_diff(img_files, imgs_train)  
    assert(len(imgs_train) == num_train)
    assert(len(imgs_val) == num_val)
    assert(len(img_files) == (len(imgs_train) + len(imgs_val)))

    # Filter dataframes
    train = df_anns[df_anns['name'].isin(imgs_train)]
    val = df_anns[df_anns['name'].isin(imgs_val)]
    assert(len(df_anns) == (len(train) + len(val)))
    print("Split data into {} train and {} val imags.".format(num_train, num_val))

    if copy_files is True:
        print("Copying image files to train and val folders...")
        src = "../images/TrafficLISA/"
        dst_train = "../images/trainTrafficLISA/"
        dst_val = "../images/valTrafficLISA/"
        count = 0
        for img in imgs_train:
            try:
                copyfile(src+img, dst_train+img)
                count += 1
            except FileNotFoundError:
                print("File {} not found in {}.".format(img, src))
        print("Copied {} files from {} to {} ({} missing).".format(count, src, dst_train, len(imgs_train) - count))
        count = 0
        for img in imgs_val:
            try:
                copyfile(src+img, dst_val+img)
                count += 1
            except FileNotFoundError:
                print("File {} not found in {}.".format(img, src))
        print("Copied {} files from {} to {} ({} missing).".format(count, src, dst_val, len(imgs_val) - count))

    return train, val


def make_coco_ann(df_ann, filename_out, save=False):
    """
    Converts a list of lists of annotations into a COCO .json object.
    """
    # Objects
    info = {
        "description": "LISA Traffic Light Dataset Subset",
        "url": "https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset",
        "version": "1.0",
        "year": 2021,
        "contributor": "Kaggle",
        "date_created": "2021/05/17"
    }

    licenses = [{
            "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "id": 9,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }]
    # Categories for the traffic dataset.
    categories = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
    {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, 
    {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
    {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
    {'supercategory': 'outdoor', 'id': 92, 'name': 'traffic_light_red'},
    {'supercategory': 'outdoor', 'id': 93, 'name': 'traffic_light_green'},
    {'supercategory': 'outdoor', 'id': 94, 'name': 'traffic_light_na'}]

    # Add images
    images = []
    img_names = list(set(df_ann['name']))

    for img_name in img_names:
        images.append({
            "license": 9,
            "file_name": img_name,
            "coco_url": "",
            "height": 960,
            "width": 1280,
            "date_captured": "2019-09-27",
            "flickr_url": "",
            "id": img_name.split('.')[0]
        })
 
    # Label mapping
    label_to_ind = {"person":1, "car":3, "bus":6, "train":7, "truck":8, "traffic light":10, "fire hydrant":11,
    "traffic_light_red":92, "traffic_light_green":93, "traffic_light_na":94}

    # Add annotations
    annotations = []
    ann_id = 1 # Format: Number + l
    df_ann.reset_index(inplace=True)

    for i in range(len(df_ann)):
        ann = df_ann.loc[i]
        name = ann['name']
        label = label_to_ind[ann['label']]
        box = [float(ann['x']), float(ann['y']), float(ann['w']), float(ann['h'])]
        annotations.append({'segmentation': [[]],
        'area': '',
        'iscrowd': 0,
        'image_id': name.split('.')[0],
        'bbox': box,
        'category_id': label,
        'id': str(ann_id) + "l"}) 
        ann_id+=1
  
    coco_ann = {'info':info, 'licenses':licenses, 'images':images, 'annotations':annotations, 'categories':categories}
    
    # Save to disk
    if save is True:
        path_ouot = "../annotations/"
        with open(str(path_ouot+filename_out)+'.json', 'w', encoding='utf-8') as f:
            json.dump(coco_ann, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename_out))

    return coco_ann


def append_coco_anns(filename_anns_1, anns_to_append, filename_out):
    """
    Appends the LISA annotations to given coco annotations.
    """
    
    # Load dataset traffic to which we want to append
    ann_file = "../annotations/" + filename_anns_1 + ".json"
    f = open(ann_file)
    anns = json.load(f)
    print("Loaded {} annotations from file {}.".format(len(anns['annotations']), filename_anns_1))

    # Append LISA stuff to the traffic dataset.
    info_out = anns['info']
    licenses_out = anns['licenses'] + anns_to_append['licenses']
    assert(len(licenses_out) == 9)
    categories_out = anns_to_append['categories']
    assert(len(categories_out) == 15)
    images_out = anns['images'] + anns_to_append['images']
    annotations_out = anns['annotations'] + anns_to_append['annotations']
    print("Number of images: {}".format(len(images_out)))
    print("Number of annotations: {}".format(len(annotations_out)))

    anns_out = {'info':info_out, 'licenses':licenses_out, 'images':images_out, 
    'annotations':annotations_out, 'categories':categories_out}
    
    # Save annotations
    with open("../annotations/" + str(filename_out)+'.json', 'w', encoding='utf-8') as f:
        json.dump(anns_out, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename_out))


if __name__ == "__main__":
    anns_lisa = load_LISA_annotations(["cocoTrafficLightsLISA-part1.csv", 
                "cocoTrafficLightsLISA-part2.csv", "cocoTrafficLightsLISA-part3.csv"])
    imgs_name_list = filter_lisa_anns(anns_lisa)
    #copy_images_from_lisa(imgs_name_list, "<path_to_the_lisa_dataset>")
    anns_lisa = anns_lisa[anns_lisa["name"].isin(imgs_name_list)]
    assert(len(set(anns_lisa['name'])) == len(imgs_name_list))

    # Split data into train and val
    df_train, df_val = split_anns(anns_lisa, split=0.8, copy_files=False)
    anns_train = make_coco_ann(df_train, "instances_trainTrafficLISA", save=False)
    anns_val = make_coco_ann(df_val, "instances_valTrafficLISA", save=False)
    append_coco_anns("./before_lisa/instances_trainTraffic", anns_train, "instances_trainTraffic")
    append_coco_anns("./before_lisa/instances_valTraffic", anns_val, "instances_valTraffic")
