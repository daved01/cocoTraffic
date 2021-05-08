
import json
import os

from pycocotools.coco import COCO

# Merge labelled traffic lights
# Make yolo dataset

def load_anns(filename):
    # Loads annotations
    path = "../annotations/" + filename
    coco=COCO(path)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds(catIds=cat_ids)
    ann_ids = coco.getAnnIds(img_ids)
    anns = coco.loadAnns(ann_ids)

    return anns, img_ids


def merge_anns(anns, anns_traffic_lights):
    """
    Args:
    anns                -- List of json annotation objects of the previous annotations.
    anns_traffic_lights -- List of json annotation objects of the new annotations.

    Returns:
    anns_merged         -- List of json annotations
    imgs_merged         -- 
    """

    # Extract annotation_ids and image_ids
    anns_ids = list(set([anns[i]['id'] for i in range(len(anns))]))
    anns_ids_traffic_lights = list(set([anns_traffic_lights[i]['id'] for i in range(len(anns_traffic_lights))]))

    img_ids = list(set([anns[i]['image_id'] for i in range(len(anns))]))
    img_ids_traffic_lights = list(set([anns_traffic_lights[i]['image_id'] for i in range(len(anns_traffic_lights))]))

    assert(len(anns_ids) == len(anns))
    assert(len(anns_ids_traffic_lights) == len(anns_traffic_lights))

    # Make final lists
    img_ids_merged = img_ids + img_ids_traffic_lights
    anns_ids_merged = anns_ids + anns_ids_traffic_lights
    assert(len(anns_ids_merged) == len(set(anns_ids_merged)))

    # Merge two annotation files.
    anns_merged = anns + anns_traffic_lights
    assert(len(anns_merged) == len(anns_ids_merged))

    return anns_merged, img_ids_merged


def get_images(img_ids):
    
    # Open file with all annotations
    filename = "../annotations/instances_train2017.json"
    coco=COCO(filename)
    cat_ids = coco.getCatIds()
    imgs = coco.loadImgs(img_ids)
    assert(len(imgs) == len(img_ids))
    print("Loaded {} image metadata!".format(len(imgs)))
    
    return imgs


def save_dataset(anns, imgs, filename):
    """
    Args:
    anns -- List of annotations json objects
    imgs -- List of image json objects
    """
    
    # Load dataset val to get structure
    ann_file = "../annotations/instances_valTrafficRelabelled.json"
    target_file = json.load(open(ann_file, 'r'))

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


def check_images():
    # Check val images
    path = "../images/valTrafficLights"
    filenames = os.listdir(path)
    filenames = [int(x.split('.')[0].lstrip('0')) for x in filenames]
    print("Found {} images in folder {}.".format(len(filenames), path))

    # Check annotation file for images
    ann_filename = "../annotations/instances_lights_val.json"
    ann_file = json.load(open(ann_filename, 'r'))
    img_ids = [ann_file['images'][i]['id'] for i in range(len(ann_file['images']))]

    # Check list of images and images in folder
    assert(len(set(filenames)) == len(set(img_ids)))
    filenames.sort()
    img_ids.sort()
    print(filenames[0:10])
    print(img_ids[0:10])
    

if __name__ == "__main__":
    # Load annotations
    """
    anns_train, _ = load_anns("instances_trainTrafficRelabelled.json") # trainTrafficRelabelled
    anns_val, _ = load_anns("instances_valTrafficRelabelled.json")
    anns_lights_train, _ = load_anns("instances_lights_train.json")
    anns_lights_val, _ = load_anns("instances_lights_val.json")

    # Merge
    anns_train_merged, imgs_train_merged = merge_anns(anns_train, anns_lights_train)
    anns_val_merged, imgs_val_merged = merge_anns(anns_val, anns_lights_val)

    imgs_train = get_images(imgs_train_merged)
    imgs_val = get_images(imgs_val_merged)

    # Save
    save_dataset(anns_train_merged, imgs_train, "trainTrafficExtended")
    save_dataset(anns_val_merged, imgs_val, "valTrafficExtended")
    """

    check_images()