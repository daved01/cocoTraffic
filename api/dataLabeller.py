from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import cv2 as cv


# Import annotations (check)
# Create loop to loop through images (check)
# Show image (check)
# Allow going backwards (check)
# Modify label category (check)
# Save modified annotations (check)
# Save progress (check)
# Save weird images (check)
# Import tags (check)
# Print progress (check)

def load_ann(filepath, saveFile):
    try:
        coco = COCO(saveFile + ".json")
        print("Previous session found. Reloading relabelled annotations")
        return coco
    except IOError:
        try:
            coco = COCO(filepath)
            print("Unable to find previous session. Starting off from original file")
            return coco
        except IOError:
            raise Exception("Could not find original file")
    


def save_tagged(target_filepath, tagged_images):
    tags = sorted(list(tagged_images))

    with open(target_filepath + ".json", 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False, indent=4)

def load_tagged(filepath):
    try:
        with open(filepath + ".json", 'r') as f:
            return set(json.load(f))
    except IOError:
        print("Unable to load tags. Starting off with all images untagged.")
        return set()

def save_point(target_filepath, imgId):
    with open(target_filepath + ".json", 'w') as f:
        json.dump(imgId, f)

def load_point(filepath, anns):
    try:
        with open(filepath + ".json", 'r') as f:
            imgId = str(json.load(f)).rstrip()
    except IOError:
        print("Unable to load last viewed image. Starting from beginning.")
        return 0

    if str(imgId) == str(-1):
        print("Previous session completed going through all annotations. Starting from beginning")
        return 0
    
    for i in range(len(anns)):
        if str(imgId) == str(anns[i]['image_id']):
            return i

    print("Unable to find matching image id in annotations. Starting from beginning")
    return 0


def box_xywh_to_xyxy(x):
    # Converts bounding boxes to (x1, y1, x2, y2) coordinates of top left and bottom right corners
    x_c, y_c, w, h = x
    # Offset for display purposes
    b = [(x_c), (y_c),
        (x_c + w), (y_c + h)]
    box = list(map(int, map(round, b)))
    return box


def save_dataset(original_filepath, target_filepath, anns, cats):

    # Load dataset val to get structure
    orig_file = json.load(open(original_filepath, 'r'))

    # Make final dictionary
    dataset = dict.fromkeys(orig_file.keys())
    dataset['info'] = orig_file['info']
    dataset['licenses'] = orig_file['licenses']
    dataset['categories'] = cats
    dataset['annotations'] = anns
    dataset['images'] = orig_file['images']

    with open(target_filepath + '.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print('Saved dataset {}.json to disk!'.format(target_filepath))


if __name__ == "__main__":
    cat_show = [10, 92, 93, 94]  # Categories ids that you want shown and relabelled
    cat_show = [10]  # Categories ids that you want shown and relabelled

    # COCO dataset path
    dataDir = ".."
    dataType = "trainTraffic"

    # Annotations file  
    annDir = "annotations"
    annFile = "{}/{}/instances_{}.json".format(dataDir, annDir, dataType)

    # Save file
    saveFile = dataDir + '/' + annDir + '/instances_' + dataType + 'Relabelled'
    tagFile = dataDir + '/' + annDir + '/instances_' + dataType + 'Tagged'
    progressFile = dataDir + '/' + annDir + '/instances_' + dataType + 'LastSave'

    # Images folder
    imgDir = dataDir + '/images/' + dataType + '/'

    # Import from annotations file
    

    coco=load_ann(annFile, saveFile)

    cats = coco.loadCats(coco.getCatIds())

    if len(cats) == 80:
        cats.append({'supercategory': 'outdoor', 'id': 92, 'name': 'traffic_light_red'})
        cats.append({'supercategory': 'outdoor', 'id': 93, 'name': 'traffic_light_green'})
        cats.append({'supercategory': 'outdoor', 'id': 94, 'name': 'traffic_light_na'})
    else:
        if cats[80]['name'] != 'traffic_light_red' or \
            cats[81]['name'] != 'traffic_light_green' or \
            cats[82]['name'] != 'traffic_light_na':
            raise Exception("Error: Categories mismatched. Check categories to make sure the 92nd category id is traffic_light_red")
    
    nms = [cat['name'] for cat in cats]
    catId_to_catName = {cats[x]['id']: cats[x]['name'] for x in range(len(cats))}

    # Load image Ids
    catIds = coco.getCatIds(catNms=nms)
    imgIdsAll = coco.getImgIds()
    imgIds = imgIdsAll
    print('Number of images: ' + str(len(imgIds)))

    # Load annotations
    annIds = coco.getAnnIds(imgIds)
    anns = coco.loadAnns(annIds)
    print('Number of annotations: ' + str(len(anns)))

    # Initialize variables
    annId_i = load_point(progressFile, anns)
    go_backwards = False
    ann_counter = 0  # To tell user how many annotations are left
    save_flag = False
    tagged_images = load_tagged(tagFile)  # (Set) of saved tagged images

    print("The available commands are as follows: (save), (q) quit, (z) back, (tag) tag image, () skip, (1)(r) label red, (2)(g) label green, (3)(n) label na, (0)(-) label back to traffic light")
    print("Type help to repeat these commands")
    
    # Main loop
    while annId_i < len(annIds):

        if annId_i < 0:
            print("You have reached the beginning")
            annId_i = 0
            go_backwards = False

        ann = anns[annId_i]
        imgId = ann['image_id']

        if ann['category_id'] in cat_show:
            go_backwards = False
            ann_counter += 1
            print()

            image = cv.imread(imgDir + (str(imgId)+'.jpg').zfill(16))
            if image is None:
                raise Exception("Error: Cannot find image {}".format(imgDir + (str(imgId)+'.jpg').zfill(16)))
            
            # Give progress status
            if ann_counter >= 10:
                print("You are on annotation {} / {}".format(str(annId_i + 1), str(len(anns))))
                ann_counter = 0

            b = ann['bbox']
            
            # Convert bounding boxes to (x1, y1, x2, y2)
            box = box_xywh_to_xyxy(b)
            print("Current label: " + catId_to_catName[ann['category_id']])

            # Display bounding box
            offset = 2  # offset bounding boxes to better see object inside
            image_bboxed = cv.rectangle(image, (box[0]-offset, box[1]-offset), (box[2]+offset, box[3]+offset), (252, 3, 219), 2)
            cv.imshow((str(imgId)+'.jpg'), image)
            if cv.waitKey(1) == ord("q"):
                break
            
            # Ask user for command
            while True:
                inp = str(input("\nAvailable commands: help, save, q, <blank>, tag, 1, 2, 3, 0, r, g, n, -\n")).rstrip().lower()
                if inp == "":
                    break
                elif inp == "help":
                    print("The available commands are as follows: (save), (q) quit, (z) back, (tag) tag image, () skip, (1)(r) label red, (2)(g) label green, (3)(n) label na, (0)(-) label back to traffic light")
                elif inp == "q":
                    if save_flag == False:
                        inp = str(input("You haven't saved. Are you sure? (y/n)\n")).rstrip().lower()
                        if inp in ['yes', 'y']:
                            print("Labels not saved")
                            exit()
                    else:
                        exit()
                elif inp == "z":
                    go_backwards = True
                    break
                elif inp == "save":
                    save_tagged(tagFile, tagged_images)
                    save_point(progressFile, imgId)
                    save_dataset(annFile, saveFile, anns, cats)
                    save_flag = True
                elif inp == "tag":
                    tagged_images.add((str(imgId)+'.jpg').zfill(16))
                    print("Added tagged image. Make sure to save to save the tag.")
                elif (inp == "r") or (inp == "1"):
                    print("Changed category id to traffic_light_red")
                    anns[annId_i]['category_id'] = 92
                    break
                elif (inp == "g") or (inp == "2"):
                    print("Changed category id to traffic_light_green")
                    anns[annId_i]['category_id'] = 93
                    break
                elif (inp == "n") or (inp == "3"):
                    print("Changed category id to traffic_light_na")
                    anns[annId_i]['category_id'] = 94
                    break
                elif (inp == "-") or (inp == "0"):
                    print("Changed category id back to traffic light")
                    anns[annId_i]['category_id'] = 10
                    break
                else:
                    print("Invalid command")

            cv.destroyAllWindows()


        # Update image and annotation indices
        if go_backwards == False:
            annId_i += 1
        else:
            annId_i -= 1


    print("Completed image labelling")
    
    # End save
    while True:
        inp = str(input("Save?(y/n)\n")).rstrip().lower()
        if inp in ['yes', 'y']:
            save_tagged(tagFile, tagged_images)
            save_point(progressFile, -1)
            save_dataset(annFile, saveFile, anns, cats)
            exit()
        elif inp in ['no', 'n']:
            inp = str(input("Are you sure?(y/n)\n")).rstrip().lower()
            if inp in ['yes', 'y']:
                print("Labels not saved")
                exit()


