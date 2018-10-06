import ssd_net
import mobilenet
import bbox_loss
import cityscape_dataset
import bbox_helper
from util import module_util
import os

current_directory = os.getcwd() #current working directory

if __name__ == '__main__':
    polygons_label_path = "cityscapes_samples_labels"
    images_path = "cityscapes_samples"
