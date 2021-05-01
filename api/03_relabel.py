# ========================================================================= #
# Merges generated new dataset splits "trainTraffic" and "valTraffic"       #
# with the previously labelled datasets to preserve the new                 #
# labels for traffic lights 92, 93, and 94.                                 #
# ========================================================================= #

import json
from pycocotools.coco import COCO


#%%
def read_annotations(ann_filename):
    # Load dataset traffic
    ann_file = "../annotations/" + ann_filename + ".json"
    f = open(ann_file)
    anns = json.load(f)
    print("Loaded {} annotations from file {}.".format(len(anns['annotations']), ann_filename))

    return anns



def read_relabelled_annotations(ann_filename, anns):
    """
    Gets annotation ids and annotations for the relabelled classes 92, 93, and 94.

    Returns:
    ann_ids -- Sorted list of annotation ids
    anns    -- List of dicts of annotations
    """
    
    # Load dataset traffic
    ann_file = "../annotations/relabelled/" + ann_filename + ".json"
    coco=COCO(ann_file)
    
    # Filter for relabelled traffic lights only
    anns_traffic_relabelled = []

    img_ids_relabelled = []
    cat_ids = [92, 93, 94]
    for cat_id in cat_ids:
        img_ids = coco.getImgIds(catIds=[cat_id])
        for img_id in img_ids:
            img_ids_relabelled.append(img_id)
    
    img_ids_relabelled = set(img_ids_relabelled)

    ann_ids_relabelled = coco.getAnnIds(img_ids_relabelled) # sorted list of ann ids
    anns_relabelled = coco.loadAnns(ann_ids_relabelled)
    for ann in anns_relabelled:
        anns_traffic_relabelled.append(ann)
    
    print('Found {} relabelled images with {} annotations for classes 92, 93, 94.'
    .format(len(img_ids_relabelled), len(anns_traffic_relabelled)))
    assert(len(ann_ids_relabelled) == len(anns_traffic_relabelled))

    return ann_ids_relabelled, anns_traffic_relabelled

  

def merge_annotations(anns, anns_relabelled, anns_ids_relabelled):
    """
    Merges relabelled annotations into annotations.

    Args:
    anns            -- Annotations json object. Keys: ['image', 'annotation', ...]
    anns_relabelled -- List of annotation
    """
    ###### RELABEL
    anns_updated = []
    count = 0

    for i in range(len(anns['annotations'])):
        ann_id = anns['annotations'][i]['id']
        
        if ann_id in anns_ids_relabelled:
            # Get annotations from relabelled
            for j in range(len(anns_relabelled)):
                temp = anns_relabelled[j]['id']
                if temp == ann_id:
                    anns_updated.append(anns_relabelled[j])
                    count = count + 1
                    continue

        else:
            anns_updated.append(anns['annotations'][i])
    
    assert(len(anns['annotations']) == len(anns_updated))

    return anns_updated



def save_annotations(anns, filename):
    """
    Saves a list of annotations to disk.
    """
    ann_file = "../annotations/instances_" + filename + ".json"
    f = open(ann_file)
    anns_out = json.load(f)
    anns_out['annotations'] = anns

    # Get updated categories
    ann_file = "../annotations/relabelled/instances_valTrafficRelabelled.json"
    f2 = open(ann_file)
    anns_out_cats = json.load(f2)
    anns_out['categories'] = anns_out_cats['categories']

    # Save to disk
    with open('../annotations/instances_'+str(filename)+'-relabelled.json', 'w', encoding='utf-8') as f:
        json.dump(anns_out, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename))



def check_annotations(anns, ann_file):
    """
    Checks if annotations have been correctly merged with the relabelled annotations.
    """
    path = "../annotations/instances_"
    coco=COCO(path+ann_file+'.json')

    # Check if annotation ids are same in both
    img_ids_org = coco.getImgIds()
    ann_ids_org = coco.getAnnIds(img_ids_org)
    anns_org = coco.loadAnns(ann_ids_org)

    assert(len(anns) == len(ann_ids_org))

    # Check the category ids in both
    for i in range(len(anns)):
        assert(anns[i]['id'] == anns_org[i]['id'])


    # Check the category ids
    cats = []
    for ann in anns:
        cats.append(ann['category_id'])
    print("Found category ids: {}".format(set(cats)))


#%% Run
if __name__ == "__main__":
    anns_train = read_annotations("instances_trainTraffic")
    anns_val = read_annotations("instances_valTraffic")

    ann_ids_relabelled_train, anns_traffic_relabelled_train = read_relabelled_annotations(
        "instances_trainTrafficRelabelled", anns_train)
    ann_ids_relabelled_val, anns_traffic_relabelled_val = read_relabelled_annotations(
        "instances_valTrafficRelabelled", anns_val)

    assert(len(ann_ids_relabelled_train) == 17940)
    assert(len(ann_ids_relabelled_val) == 5307)

    # New datasets contain same images as previoulsy relabelled one due to seed, but they could be 
    # allocated to different sets due to different seeds for the splits.
    ann_ids_relabelled = ann_ids_relabelled_train + ann_ids_relabelled_val
    anns_traffic_relabelled = anns_traffic_relabelled_train + anns_traffic_relabelled_val
    assert(len(ann_ids_relabelled) == (17940 + 5307))

    # Merge relabelled annotations into annotations
    anns_train = merge_annotations(anns_train, anns_traffic_relabelled, ann_ids_relabelled)
    anns_val = merge_annotations(anns_val, anns_traffic_relabelled, ann_ids_relabelled)

    save_annotations(anns_train, "trainTraffic")
    save_annotations(anns_val, "valTraffic")

    # Check structure of produced datasets
    check_annotations(anns_train, "trainTraffic")
    check_annotations(anns_val, "valTraffic")
