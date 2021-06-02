# ========================================================================= #
# Generates the subsets for the COCO dataset with refined traffic light     #
# class. New traffic light classes are:                                     #
# 92 traffic_light_red                                                      #
# 93 traffic_light_green                                                    #
# 94 traffic_light_na                                                       #
# Mislabelled COCO data keep their class traffic light (10)                 #
#                                                                           #
# Inputs:                                                                   #
# - COCO train2017                                                          #
# - COCO val2017                                                            #
# - LISA Traffic Light Dataset                                              #
#                                                                           #
# Outputs:                                                                  #
# 0. Base annotations - Refined traffic lights from train2017, val2017      # 
# 1. COCO Refined - train2017 and val2017, traffic lights refined           #
# 2. COCO Traffic - subsampled train2017 and traffic lights form val2017    #
# 3. COCO Traffic Extended - Like 2. with added images from the LISA        #
#                            Traffic Light dataset                          #
# ========================================================================= #

import json
import random
from shutil import copyfile
from collections import defaultdict


def load_anns(path, filename): 
    # Loads an annotation file
    f = open(path+filename)
    anns = json.load(f)

    return anns

def save_dataset(dataset, filename):
    # Saves an annotation file
    path = "../annotations/"
    with open(path+filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=0)
    print("Saved dataset to disk as {}!".format(filename))

def get_diff(l1, l2):
    # Returns the difference between two lists.
    diff = list( list(set(l1) - set(l2)) + list(set(l2) - set(l1)) )
    return diff

def filter_classes(dataset):
    """
    Filters classes from the dataset based on class_names.
    Inputs:
    dataset - COCO annotation file object

    Returns:
    dataset - COCO annotation file objecct with specified classes only.
    """
    class_names = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 
                    'bicycle', 'person', 'dog', 'cat', 'stop sign', 
                    'fire hydrant', 'train', 'traffic_light_red', 
                    'traffic_light_green', 'traffic_light_na']
    class_names = ['traffic light', 'traffic_light_red', 
                    'traffic_light_green', 'traffic_light_na']
    
    # Get category ids
    keep = []
    for cat in dataset['categories']:
        if cat['name'] in class_names:
            keep.append(int(cat['id']))
    assert(len(keep) == len(class_names))
  
    # Map img_ids to image objects
    images_table = {x['id']:x for x in dataset['images']}

    # Map img_ids to annotation objects
    img_id_table = defaultdict(list)

    for ann in dataset['annotations']:
        if ann['category_id'] in keep:
            img_id_table[ann['image_id']].append(ann)
    
    print("Removed {} / {} images by category.".format(len(dataset['images']) - len(img_id_table.keys()), len(dataset['images']) ))

    imgs_out = []
    imgs_ids_out = []
    anns_out = []

    for img_id in img_id_table:
        imgs_ids_out.append(int(img_id))
        anns_out += img_id_table[img_id]

    for img_id in imgs_ids_out:
        imgs_out.append(images_table[img_id])

    dataset['images'] = imgs_out
    dataset['annotations'] = anns_out

    # Check output
    assert(len(dataset['images']) == len(set(ann['image_id'] for ann in dataset['annotations'])))

    return dataset

def copy_image_files(img_ids, foldername):
    """
    Copies images based on their image_ids into the specified folder.
    """
    
    # Load all traffic related image ids:
    anns = load_anns("/Users/David/Repositories/cocoTraffic/annotations/21_coco_sub_all_traffic/", 
    'instances_val2017Relabelled.json')   
    
    # Filter for traffic lights:
    path = '../images/'
    count = 0

    for img in img_ids:
        filename = (str(img) + ".jpg").zfill(16)

        try:
            copyfile(path+'val2017/'+filename, path+foldername+'/'+filename)
            count +=1
        except FileNotFoundError:
            print('Folder {} does not exist. Please create it and try again.'.format(path+foldername))
            return
    assert(count == len(img_ids))

    print('Copied {} images to {}'.format(count, path+foldername))

def make_base_dataset(anns_train1, anns_train2, anns_val):
    """
    Returns all labelled traffic light annotations from the three annotation files.
    """

    keys = ['info', 'licenses', 'images', 'annotations', 'categories']
    anns = dict.fromkeys(keys)
    
    # Append basics from train1
    anns['info'] = anns_train1['info']
    anns['licenses'] = anns_train1['licenses']
    anns['categories'] = anns_train1['categories']

    # Get images
    target_classes = [10,92,93,94]

    anns_out = []
    anns_in = anns_train1['annotations'] + anns_train2['annotations'] + anns_val['annotations']
    
    print("Found {} annotations in total.".format(len(anns_in)))
    anns_count = {10:0, 92:0, 93:0, 94:0}
    seen = []

    for ann in anns_in:
        if (int(ann['category_id']) in target_classes) and (ann['id'] not in seen):
            anns_out.append(ann)
            seen.append(ann['id'])
            anns_count[int(ann['category_id'])] += 1
    print("Filtered {} annotations containing traffic lights.".format(len(anns_out)))

    images_out =[]
    images_in = anns_train1['images'] + anns_train2['images'] + anns_val['images']
    images_keep = [int(x['image_id']) for x in anns_out]
    print("Found {} images in total. Keeping {} images.".format(len(images_in), len(images_keep)))
    
    for image in images_in:
        if int(image['id']) in images_keep:
            images_out.append(image)

    print("Added {} images containing traffic lights.".format(len(images_out)))

    # Add both to anns
    anns['annotations'] = anns_out
    anns['images'] = images_out

    return anns

def make_coco_refined(dataset_in, dataset_relabelled):
    """
    Replaces traffic light annotations with the relabelled ones.

    Inputs:
    dataset_in         - COCO dataset object, e.g. train2017, val2017

    Returns:
    dataset_relabelled - COCO dataset object
    """
    
    # Only difference lays in some annotations.
    keys = ['info', 'licenses', 'images', 'annotations', 'categories']
    dataset = dict.fromkeys(keys)
    dataset['info'] = dataset_in['info']
    dataset['licenses'] = dataset_in['licenses']
    dataset['categories'] = dataset_relabelled['categories']
    dataset['images'] = dataset_in['images']
    
    anns_change = dict()
    for ann in dataset_relabelled['annotations']:
        ann_id = ann['id']
        anns_change[ann_id] = ann

    anns_out = []
    count = 0
    count_total = 0
    for ann in dataset_in['annotations']:
        if ann['id'] in anns_change:
            count += 1
            anns_out.append(anns_change[int(ann['id'])])
        else:
            anns_out.append(ann)
            count_total += 1

    dataset['annotations'] = anns_out
   
    print("Found {} / {} relabelled annotations.".format(count, count_total+count))
    assert(len(anns_out) == len(dataset_in['annotations']))
    assert(count+count_total == len(dataset_in['annotations']))
    assert(len(dataset['images']) == len(dataset_in['images']))
    assert(len(dataset_in['annotations']) == len(dataset['annotations']))
    
    return dataset

def make_coco_traffic(dataset_train, dataset_val, dataset_add):
    """
    Takes the new datast dataset_add, splits it into train/val 80/20
    and appends this to the other datasets.
    """
    # Prepare output
    num_imgs_train_in = len(dataset_train['images'])
    num_imgs_val_in = len(dataset_val['images'])

    images_add = [x['id'] for x in dataset_add['images']]
    print("Found {} images which will be split into train and val.".format(len(images_add)))

    images_table = {x['id']:x for x in dataset_add["images"]}

    # Image_id to list of annotation object
    anns_table = defaultdict(list)
    for ann in dataset_add['annotations']:
        img_id = ann['image_id']
        anns_table[img_id].append(ann)

    split = 0.8
    random.seed(1881)
    num_train = int(len(images_add) * split)
    num_val = len(images_add) - num_train
    imgs_train = random.sample(images_add, num_train)
    imgs_val = get_diff(images_add, imgs_train)

    assert(len(imgs_train) == num_train)
    assert(len(imgs_val) == num_val)
    assert(len(images_add) == (num_train + num_val))

    images_train_out = [x for x in dataset_train['images']]
    images_val_out = [x for x in dataset_val['images']]
    anns_train_out = [x for x in dataset_train['annotations']]
    anns_val_out = [x for x in dataset_val['annotations']]

    for img_id in imgs_train:    
        images_train_out.append(images_table[img_id])
        anns_train_out += anns_table[img_id]

    for img_id in imgs_val:
        images_val_out.append(images_table[img_id])
        anns_val_out += anns_table[img_id]
    
    dataset_train['images'] = images_train_out
    dataset_train['annotations'] = anns_train_out
    dataset_val['images'] = images_val_out
    dataset_val['annotations'] = anns_val_out

    assert(len(dataset_train["images"]) == (num_imgs_train_in + num_train))
    assert(len(dataset_val["images"]) == (num_imgs_val_in + num_val))

    return dataset_train, dataset_val, imgs_train, imgs_val
    
def make_coco_traffic_extended(dataset_in, dataset_append):
    """
    Appends the traffic light annotations from val2017 to dataset_in.
    """
    keys = ['info', 'licenses', 'images', 'annotations', 'categories']
    dataset_out = dict.fromkeys(keys)

    dataset_out['info'] = dataset_in['info']
    dataset_out['licenses'] = dataset_in['licenses'] + dataset_append['licenses']
    assert(len(dataset_out['licenses']) == 9)
    dataset_out['categories'] = dataset_in['categories']
    dataset_out['images'] = dataset_in['images'] + dataset_append['images']
    dataset_out['annotations'] = dataset_in['annotations'] + dataset_append['annotations']

    assert(len(dataset_out['images']) == (len(dataset_in['images']) + len(dataset_append['images'])))
    assert(len(dataset_out['annotations']) == (len(dataset_in['annotations']) + len(dataset_append['annotations'])))

    return dataset_out

def make_new_images(dataset, imgs_train, imgs_val):
    """
    Split the annotations in dataset into two files train and val
    according to the img ids in imgs_train, imgs_val.
    """
    table_imgs = {x['id']:x for x in dataset['images']}
    table_anns = {x['image_id']:x for x in dataset['annotations']}

    keys = ['info', 'licenses', 'images', 'annotations', 'categories']
    
    # Train
    dataset_train = dict.fromkeys(keys)
    dataset_train['info'] = dataset['info']
    dataset_train['licenses'] = dataset['licenses']
    dataset_train['categories'] = dataset['categories']
    dataset_train['images'] = [table_imgs[x] for x in imgs_train]
    dataset_train['annotations'] = [table_anns[x] for x in imgs_train]

    # Validation
    dataset_val = dict.fromkeys(keys)
    dataset_val['info'] = dataset['info']
    dataset_val['licenses'] = dataset['licenses']
    dataset_val['categories'] = dataset['categories']
    dataset_val['images'] = [table_imgs[x] for x in imgs_val]
    dataset_val['annotations'] = [table_anns[x] for x in imgs_val]
    
    return dataset_train, dataset_val

def print_stats(dataset):
    """
    Prints stats for the dataset.
    Number of images
    Number of annotations total
    Number of annotations per object class (dict)
    """
    print("\n================================================\nStats\n================================================")
    num_images = len(dataset["images"])
    num_anns_total = len(dataset["annotations"])
    print("Number of images: {}\nNumber of annotations: {}\n".format(num_images, num_anns_total))

    anns_per_class = dict()
    for ann in dataset["annotations"]:
        cl = ann["category_id"]
        if cl in anns_per_class:
            anns_per_class[cl] += 1
        else:
            anns_per_class[cl] = 1
    print(anns_per_class)     


if __name__ == "__main__":
    # 0. Dataset: COCO Traffic Lights
    path = "../annotations/21_coco_sub_all_traffic/"
    file_train1 = "instances_trainTraffic.json"
    file_train2 = "instances_valTraffic.json"
    file_val = "instances_val2017Relabelled.json"
    anns_train1 = load_anns(path, file_train1)
    anns_train2 = load_anns(path, file_train2)
    anns_val = load_anns(path, file_val)
    dataset1 = make_base_dataset(anns_train1, anns_train2, anns_val)
    #save_dataset(dataset1, "instances_traffic_lights.json")

    # 1. Dataset: COCO Refined
    print("----------\nDataset 1")
    path = "../annotations/"
    anns_relabelled = dataset1
    file_train = "instances_train2017.json"
    file_val = "instances_val2017.json"
    anns_train = load_anns(path, file_train)
    anns_val = load_anns(path, file_val)
    dataset2train = make_coco_refined(anns_train, anns_relabelled)
    dataset2val = make_coco_refined(anns_val, anns_relabelled)
    #save_dataset(dataset2train, "instances_train2017refined.json")
    #save_dataset(dataset2val, "instances_val2017refined.json")

    # 2. Dataset: COCO Traffic
    print("----------\nDataset 2")
    path = "../annotations/21_coco_sub_all_traffic/"
    file_train = "instances_trainTraffic.json"
    file_val = "instances_valTraffic.json"
    anns_train = load_anns(path, file_train)
    anns_val = load_anns(path, file_val)
    path_add = "../annotations/21_coco_sub_all_traffic/"
    file_val_add = "instances_val2017Relabelled.json"
    anns_add = load_anns(path_add, file_val_add)
    print(len(anns_add['annotations']))
    anns_add = filter_classes(anns_add)
    print(len(anns_add['annotations']))
    print_stats(anns_train)
    print_stats(anns_val)
    print_stats(anns_add)
    train_out, val_out, imgs_train, imgs_val = make_coco_traffic(anns_train, anns_val, anns_add)
    dataset_train, dataset_val = make_new_images(anns_add, imgs_train, imgs_val)
    #save_dataset(dataset_train, "instances_train_new_images.json")
    #save_dataset(dataset_val, "instances_val_new_images.json")

    print("\n############## Modified dataset 2 ##############")
    print_stats(train_out)
    print_stats(val_out)
    #save_dataset(train_out, "instances_train_traffic.json")
    #save_dataset(val_out, "instances_val_traffic.json")
    #copy_image_files(imgs_train, "trainTraffic")
    #copy_image_files(imgs_val, "valTraffic")

    # 3. Dataset: COCO Traffic Extended
    print("----------\nDataset 3")
    path = "../annotations/30_lisa_sub/"
    file_train = "instances_trainTrafficLISA.json"
    file_val = "instances_valTrafficLISA.json"
    train_append = load_anns(path, file_train)
    val_append = load_anns(path, file_val)
    dataset_train = make_coco_traffic_extended(train_out, train_append)
    dataset_val = make_coco_traffic_extended(val_out, val_append)
    print_stats(dataset_train)
    print_stats(dataset_val)
    #save_dataset(dataset_train, "instances_train_traffic_extended.json")
    #save_dataset(dataset_val, "instances_val_traffic_extended.json")
    