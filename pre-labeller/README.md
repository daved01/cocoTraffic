# COCO Pre labeller
Takes an input images and predicts bounding boxes for the COCO dataset. These can be relabelled, e.g. `traffic light` can be refined into `traffic light red` and `traffic light green` by changing the labels. This significanlty reduces the effort to generate more such data.
It uses a DETR model.

## Files
`make_annotations.py`
Takes images and predicts bounding boxes and COCO labels. Writes results to a `.csv` file.

`make_coco_annotations.py`
Takes predictions from `make_annotations.py` in `.csv` format and transforms them into a COCO-style `.json` annotation file. Metadata like licenses etc. are hard-coded for the [LISA Traffic Light Dataset](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset) and have to be adjusted for other datasets.


## Usage
Type `python make_annotations.py` and enter the filename of a `.txt` file with a list of image filepaths for which you want to predict bounding boxes.
The file must have one path per line.

The output is a file `annotations.csv` which has the columns `image name`, `COCO label`, and the COCO bounding box coordinates `(x1, y1)` (upper left) and `(x2, y2)` (lower right).