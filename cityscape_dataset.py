import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors


class CityScapeDataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        # TODO: implement prior bounding box
        """
            Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

            Use VGG_SSD 300x300 as example:
            Feature map dimension for each output layers:
               Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
            1. Conv11    | (19x19)        | (60x60) (unit. pixels)
            2. Conv13    | (10x10)        | (113x113)
            3. Conv14_2  | (5x5)          | (165x165)
            4. Conv15_2  | (3x3)          | (218x218)
            5. Conv16_2  | (1x1)          | (270x270)
            6. Conv17_2  | (1x1)          | (264x264)
            NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
            Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
            :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
            :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
            """
        prior_layer_cfg = [{'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                           {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (113, 113),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                           {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (165, 165),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                           {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (218, 218),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                           {'layer_name': 'Conv16_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                           {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
                           ]
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg)

        # Pre-process parameters:
        #  Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        sample_labels = None
        sample_bboxes = None
        sample_img = None

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, sample_bboxes, sample_labels,
                                                      iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_tensor, bbox_label_tensor
