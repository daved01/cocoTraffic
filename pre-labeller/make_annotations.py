# =================================================================== #
# Program to pre-annotate images using COCO-trained DETR.             #
#                                                                     #
# Input:                                                              #
# File with paths to each image which should be annotated.            #
#                                                                     #
# Output:                                                             #
# Annotation file in the COCO format                                  #
# Box format: (x, y, w, h), where x,y is the top left corner.         #
                                                                      #
# Note that this does not work well when the image is crowed.         #
# =================================================================== #

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import csv


def read_list_to_annotate(filename):
    """
    Reads the file containing the paths to the images.
    Textfile
    """
    img_paths = []
    with open(filename, "r") as f:
        for line in f:
            img_paths.append(str(line).rstrip('\n'))
    print("Loaded names of {} images.".format(len(set(img_paths))))
    return img_paths


def box_cxcywh_to_xywh(x):
    # Converts bounding boxes to (x1, y1, w, h) coordinates of top left corner and width and height.

    # (center_x, center_y, h, w)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        w, h]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    # Scale the bounding boxes to the image size
    img_w, img_h = size
    b = box_cxcywh_to_xywh(out_bbox)
    #b = out_bbox
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def auto_annotate(img_paths):
    """
    Auto annotates a list of images using DETR.
    Args:

    """
    detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    detr.eval()

    annotations = []
    for img_path in img_paths:
        res = predict(detr, img_path)
        annotations += res
        
    return annotations


def predict(model, img_path, thresh= 0.6):
    """
    Predicts for a given image path
    """
    
    # Load image from path
    filename = img_path.split('/')[-1]
    img = Image.open(img_path)

    # Preprocess image
    transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Predict
    t_image = transform(img).unsqueeze(0)
    output = model(t_image)
    probas = output['pred_logits'].softmax(-1)[0,:,:-1]

    boxes = rescale_bboxes(output['pred_boxes'][0], (img.size[0], img.size[1])).detach()
    labels = probas.max(-1).indices
    conf = probas.max(-1).values.detach()
    
    # Threshold scores
    conf = conf.detach()
    keep = conf > thresh

    # Filter out scores, boxes, and labels using threshold
    conf = conf[keep]
    boxes = boxes.detach()[keep].numpy()
    labels = labels.detach()[keep].numpy()

    print("Predicted {} annotations for image {}...".format(len(labels), filename))
    # Output: file, label, box1, box2, box3, box4
    out = []
    for i in range(len(labels)):
        out.append([filename, labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])
    
    return out


def save_annotations(anns):
    filename_out = 'annotations.csv'
    file = open(filename_out, 'w+', newline = '')

    with file:
        write = csv.writer(file)
        write.writerows(anns)

    print("Saved annotations to {}!".format(filename_out))


if __name__ == "__main__":
    filename = input("Enter name of file with annotations: ")
    if filename == "":
        filename = 'to_annotate-test.txt'
        print("Using default name {}".format(filename))
    img_paths = read_list_to_annotate(filename)
    anns = auto_annotate(img_paths)
    save_annotations(anns)