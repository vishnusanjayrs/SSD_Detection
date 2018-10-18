import numpy as np
import torch.nn
from torch.utils.data import Dataset
from bbox_helper import generate_prior_bboxes, match_priors
from PIL import Image
from random import randint
import matplotlib.pyplot as plt


class CityScapeDataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.image_size = 300

        # TODO: implement prior bounding box
        """
            Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

            Use VGG_SSD 300x300 as example:
            Feature map dimension for each output layers:
               Layer     | Map Dim (h, w)  | Single bbox size that covers in the original image
            1. Conv6     | (38x38)         | (30x30) (unit. pixels)
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
        prior_layer_cfg = [{'layer_name': 'Conv4', 'feature_dim_hw': (75, 75), 'bbox_size': (7.5, 7.5),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv6', 'feature_dim_hw': (38, 38), 'bbox_size': (40.71, 40.71),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (73.93, 73.93),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (107.14, 107.14),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (140.36, 140.36),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (173.57, 173.57),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (206.79, 206.79),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)},
                           {'layer_name': 'Conv16_2', 'feature_dim_hw': (1, 1), 'bbox_size': (240, 240),
                            'aspect_ratio': (1.0, 1 / 2, 1 / 3, 1 / 4, 2.0, 3.0, 4.0)}
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
        item = self.dataset_list[idx]
        image_path = item['image_path']
        labels = np.asarray(item['labels'])
        labels = torch.Tensor(labels).cuda()
        bboxes = torch.Tensor(item['bboxes']).cuda()
        bbox = np.array(item['bboxes'])
        # print(self.bboxes.shape)

        img = Image.open(image_path)
        #
        # plt.imshow(img)
        # plt.show()

        # img = self.crop(img, bbox, labels)

        # print('img_size',img.size)

        w, h = img.size
        bboxes /= torch.Tensor([w, h, w, h]).expand_as(bboxes)

        # resize image
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)

        # plt.imshow(img)
        # plt.show()
        # normalize_img
        img = np.asarray(img, dtype=np.float32)
        # normalise the image pixels to (-1,1)
        img = np.subtract(img, self.mean)
        img = np.divide(img, self.std)

        # convert to tensor
        img_tensor = torch.Tensor(img.astype(float))
        img_tensor = img_tensor.view((img.shape[2], img.shape[0], img.shape[1]))
        img_tensor = img_tensor.cuda()

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, bboxes, labels,
                                                      iou_threshold=0.5)

        # [DEBUG] check the output.
        # assert isinstance(bbox_label_tensor, torch.Tensor)
        # assert isinstance(bbox_tensor, torch.Tensor)
        # assert bbox_tensor.dim() == 2
        # assert bbox_tensor.shape[1] == 4
        # assert bbox_label_tensor.dim() == 1
        # assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]




        return img_tensor, bbox_tensor, bbox_label_tensor

    def crop(self,image, bbox, label):
        w, h = image.size
        print(bbox)
        print('image_size insde crop :',w,h)
        xmin = min(bbox[:, 0])
        xmax = max(bbox[:, 2])
        ymin = min(bbox[:, 1])
        ymax = max(bbox[:, 3])
        print('xmin', xmin)
        print('xmax', xmax)
        if xmax - xmin < h:
            print("image within range")
            max_dim = max((xmax - xmin),(ymax-ymin))
            size = randint(max_dim,h)
            print(size)
            add_x = size - (xmax - xmin)
            add_y = size - (ymax - ymin)
            rand = np.random.dirichlet(np.ones(2), size=1)
            dec_l = round(add_x * rand[0, 0])
            inc_r = round(add_x * rand[0, 1])
            dec_t = round(add_y * rand[0, 0])
            inc_b = round(add_y * rand[0, 1])
            if xmin - dec_l < 0:
                l=0
                r=size


            crops = np.array([l, t, r, b])
            print(crops)
            crop_img = image.crop(crops)
            return crop_img
        else:
            return image
