# ========================================================================= #
# Adds remaining traffic light images from COCO to another dataset.         #
# Inputs: Annotation files in the COCO format for both datasets             #
#                                                                           #
# ========================================================================= #

import random
import json
import csv
from shutil import copyfile

from pycocotools.coco import COCO


# Read traffic lights
def get_diff(l1, l2):
    """
    Returns the difference between two lists.
    """
    diff = list( list(set(l1) - set(l2)) + list(set(l2) - set(l1)) )

    return diff


def get_image_ids(filename):
    """"
    Gets image ids of images which we want to sample.
    """
    # Load annotations
    coco=COCO(filename)
    
    # Sample traffic light images
    print("-----------\nSampling traffic light images...")
    catIds = coco.getCatIds(catNms='traffic light')
    assert(catIds == [10])
    img_ids = coco.getImgIds(catIds=catIds)
    print('Total number of images: {}'.format(len(img_ids)))

    return img_ids


def split_data(img_ids, split_train=.8):
    """
    Splits the img_ids into train and val.
    """
    num_train = int(len(img_ids) * split_train)
    num_val = len(img_ids) - num_train

    train = random.sample(img_ids, num_train)
    val = get_diff(img_ids, train)

    assert(len(img_ids) == len(train) + len(val))

    return train, val


def get_annotations(filename, img_ids):
    """
    Gets annotations for the image ids.
    """
    coco=COCO(filename)

    ann_ids = coco.getAnnIds(img_ids)
    anns = coco.loadAnns(ann_ids) # List of dictionaries for each annotation
    print('Number of object instances in the {} images: {}'.format(len(img_ids), len(anns)))

    # Remove other classes from data
    print("\nRemoving out of scope classes...")
    class_names = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'dog', 
                'cat', 'stop sign', 'fire hydrant', 'train']
    anns_cleaned = []

    for ann in anns:
        if ann['category_id'] in coco.getCatIds(catNms=class_names):
            anns_cleaned.append(ann)
            
    print('Removed {} annototations from the data.\nNumber of images: {}\nNumber of annotations: {}'
            .format(len(anns)-len(anns_cleaned), 
    len(img_ids), len(anns_cleaned)))

    # Check
    img_ids_anns = list(set([x['image_id'] for x in anns_cleaned]))
    assert(len(img_ids) == len(img_ids_anns))
    img_ids.sort()
    img_ids_anns.sort()
    assert((img_ids == img_ids_anns) is True)

    return anns_cleaned


def save_dataset(img_ids, anns, filename):
    # Load dataset val to get structure
    ann_file = "../annotations/instances_train2017.json"
    target_file = json.load(open(ann_file, 'r'))
    coco=COCO(ann_file)
    imgs = coco.loadImgs(img_ids)

    # Check
    img_ids_from_imgs = list(set([x['id'] for x in imgs]))
    img_ids_anns = list(set([x['image_id'] for x in anns]))
    assert(len(img_ids_from_imgs) == len(img_ids_anns))
    img_ids_from_imgs.sort()
    img_ids_anns.sort()
    assert((img_ids_from_imgs == img_ids_anns) is True)
    
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
    assert(count == len(img_ids))
    print('Copied {} images to {}'.format(count, path+foldername))
   


if __name__ == "__main__":
    img_ids_all = get_image_ids("../annotations/instances_train2017.json")
    img_ids_labelled = get_image_ids("../annotations/split_before_relabelling/instances_traffic.json")
    
    # Get image ids to be sampled
    img_ids_sampled = get_diff(img_ids_all, img_ids_labelled) # About 2000 additional images
    assert(len(img_ids_all) == len(img_ids_labelled) + len(img_ids_sampled))

    # Split data
    train, val = split_data(img_ids_sampled)

    # Get annotations
    anns_train = get_annotations("../annotations/instances_train2017.json", train)
    anns_val = get_annotations("../annotations/instances_train2017.json", val)

    # Save additional annoations
    save_dataset(train, anns_train, "lights_train")
    save_dataset(val, anns_val, "lights_val")

    # Copy images. Keep separate so they can easily be uploaded.
    copy_image_files(train, "trainTrafficLights")
    copy_image_files(val, "valTrafficLights")
