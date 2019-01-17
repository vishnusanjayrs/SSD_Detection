import torch
import math
import numpy as np

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''
img_h = 300
img_w = 600


def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv4    | (38x38)        | (30x30) (unit. pixels)
    2. Conv7    | (19x19)        | (60x60)
    3. Conv8_2  | (10x10)        | (111x111)
    4. Conv9_2  | (5x5)          | (162x162)
    5. Conv10_2 | (3x3)          | (213x213)
    6. Conv11_2 | (1x1)          | (264x264)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """
    example_prior_layer_cfg = [
        # Example:
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60),
         'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
        {'layer_name': 'Conv4', 'feature_dim_hw': (64, 64), 'bbox_size': (60, 60),
         'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
        # ...
        # TODO: define your feature map settings
    ]

    priors_bboxes = []
    # init k+1 bbox size to avoid error

    for feat_level_idx in range(0, len(prior_layer_cfg)):  # iterate each layers
        # print("feat_level_idx")
        # print(feat_level_idx)
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']
        bbox_dim = layer_cfg['bbox_size']

        # Todo: compute S_{k} (reference: SSD Paper equation 4.)
        sk = bbox_dim[0] / img_h
        if feat_level_idx == len(prior_layer_cfg) - 1:
            # print("skplus1 here")
            skplus1 = 1.04
        else:
            layer_cfgplus1 = prior_layer_cfg[feat_level_idx + 1]
            bbox_plus1 = layer_cfgplus1['bbox_size']
            skplus1 = bbox_plus1[0] / img_h

        fk = layer_feature_dim[0]
        for y in range(0, layer_feature_dim[0]):
            for x in range(0, layer_feature_dim[0]):

                # Todo: compute bounding box center
                cx = (x + 0.5) / fk
                cy = (y + 0.5) / fk

                h = sk
                w = sk
                priors_bboxes.append([cx, cy, w, h])

                h = math.sqrt(sk * skplus1)
                w = math.sqrt(sk * skplus1)
                priors_bboxes.append([cx, cy, w, h])

                # Todo: generate prior bounding box with respect to the aspect ratio
                for aspect_ratio in layer_aspect_ratio:
                    h = sk / math.sqrt(aspect_ratio)
                    w = sk * math.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])
                    priors_bboxes.append([cx, cy, h, w])

    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    num_priors = priors_bboxes.shape[0]
    # print(num_priors)
    # print(priors_bboxes.dim)

    # [DEBUG] check the output shape
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes.cuda()


def intersect_ios(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection.
    :param a: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: intersections values: dim: (n_item).
    """
    # Compute the intersections recangle.
    rec_a = center2corner(a)
    rec_b = center2corner(b)
    intersections = torch.cat((torch.max(rec_a, rec_b)[..., :2], torch.min(rec_a, rec_b)[..., 2:]), -1)

    # Compute the intersections area.
    x1 = intersections[..., 0]
    y1 = intersections[..., 1]
    x2 = intersections[..., 2]
    y2 = intersections[..., 3]
    sub1 = torch.sub(x2, x1)
    sub2 = torch.sub(y2, y1)
    sub1[sub1 < 0] = 0
    sub2[sub2 < 0] = 0

    intersections = torch.mul(sub1, sub2)

    return intersections


def intersect(a_box, b_box):
    # compute intersection area between 2 bounding boxes
    # inpputs should be in corner form

    # get size of input boxes
    A = a_box.size(0)
    B = b_box.size(0)

    # compute intersection points between 2 rectangles in both dimensions
    # expand for size of input bboxes to get all intersects of all combinations
    max_xy = torch.min(a_box[:, 2:].unsqueeze(1).expand(A, B, 2),
                       b_box[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(a_box[:, :2].unsqueeze(1).expand(A, B, 2),
                       b_box[:, :2].unsqueeze(0).expand(A, B, 2))

    # get the intersection rectangle dimensions and clamp negative values
    inter = torch.clamp((max_xy - min_xy), min=0)

    # calculate area
    intersect_area = inter[:, :, 0] * inter[:, :, 1]

    return intersect_area


def area(box):
    # area of rectangle
    box_area = ((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))

    return box_area


def preprocess(a: torch.Tensor, b: torch.Tensor):
    """
    # Preprocess the tensor so both input will be in the same size.
    :param a: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference.
    :return: bounding box tensors with same dimension.
    """
    if a.shape == b.shape:
        return a, b
    if b.shape == torch.Size([4]):
        b = b.unsqueeze(0)
    b = b.repeat(a.shape[0], 1)

    return a, b


def ios(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the intersection over smaller object.
    :param a: area, dim: (n_items, 4).
    :param b: area, dim: (n_items, 4).
    :return: intersection over smaller value: dim: (n_items).
    """
    a, b = preprocess(a, b)
    intersections = intersect_ios(a, b)
    area_a = torch.mul(a[..., 2], a[..., 3])
    area_b = torch.mul(b[..., 2], b[..., 3])

    return torch.div(intersections, torch.min(area_a, area_b))


def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4)
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    assert a.dim() == 2
    assert a.shape[1] == 4
    assert b.dim() == 2
    assert b.shape[1] == 4

    # IoU of two bounding box
    # area (A union B) = area(A) + area(B) = area(A intersect B)
    inter = intersect(a, b)
    a_area = area(a).unsqueeze(1).expand_as(inter)
    b_area = area(b).unsqueeze(0).expand_as(inter)
    union = a_area + b_area - inter
    iou = inter / union

    # [DEBUG] Check if output is the desire shape
    assert iou.dim() == 2
    assert iou.shape[0] == a.shape[0]
    return iou


def match_priors(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your ''cityscape_dataset.py', see the SSD paper page 5 for reference. (note that default box = prior boxes)

    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    # print("In match_priors ")
    # get iou between al ground truth and prior boxes
    gtpr_iou = iou(gt_bboxes, center2corner(prior_bboxes))

    # get maximun iou and index of groundtruth for each prior box
    iou_val, max_idx = gtpr_iou.max(0)
    max_idx.squeeze_(0)
    iou_val.squeeze_(0)
    matched_boxes = gt_bboxes[max_idx]

    # encode variances and convert into locations
    variances = [0.1, 0.2]
    cxcy = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2 - prior_bboxes[:, :2]  # [8732,2]
    cxcy /= variances[0] * prior_bboxes[:, 2:]
    wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / prior_bboxes[:, 2:]  # [8732,2]
    wh = torch.log(wh) / variances[1]

    loc = torch.cat([cxcy, wh], 1)

    # encode the labels and set the labels to zero where iou is less than threshold
    matched_labels = gt_labels[max_idx]
    matched_labels[iou_val < iou_threshold] = 0  # using iou_threshold to set background
    loc[iou_val < iou_threshold] = 0.0

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return loc, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''


def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.9):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = []

    # convert bboxes from center format to corner format
    num_classes = bbox_confid_scores.shape[1]
    for class_idx in range(0, num_classes):
        # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast
        # computation
        # filtering scores using probability threshold
        bbox_loc_c = bbox_loc
        print(class_idx)
        if class_idx == 0:  # ignoring background case
            continue
        bbx_class_scores = bbox_confid_scores[:, class_idx]
        print(bbx_class_scores.shape)
        filtered_pos = bbx_class_scores > prob_threshold
        prob_fil_scores = bbx_class_scores[filtered_pos]
        bbox_loc_c = bbox_loc_c[filtered_pos,:]
        print(prob_fil_scores.shape)
        if filtered_pos.data.sum() == 0:
            continue

        pick = []
        l = bbox_loc_c[:, 0]
        t = bbox_loc_c[:, 1]
        r = bbox_loc_c[:, 2]
        b = bbox_loc_c[:, 3]

        # calculating area
        areas = (r - l) * (b - t)
        sorted_scores, order = prob_fil_scores.sort(0, descending=True)

        print(sorted_scores, order)

        while order.numel() > 0:
            i = order[0]
            pick.append(int(i))

            if order.numel() == 1:
                break

            xx1 = l[order[1:]].clamp(min=l[i])
            yy1 = t[order[1:]].clamp(min=t[i])
            xx2 = r[order[1:]].clamp(max=r[i])
            yy2 = b[order[1:]].clamp(max=b[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            ids = (ovr <= overlap_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        sel_bbox.append(bbox_loc_c[pick])


    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''


def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w).
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,  # b_{center}
        p_size * torch.exp(size_var * l_size)  # b_{size}
    ], dim=-1)


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form.
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)


def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:] / 2,
                      center[..., :2] + center[..., 2:] / 2], dim=-1)


def corner2center(corner):
    """
    Convert bounding box from corner form (x,y) (x+w, y+h) to  center form (cx, cy, w, h)
    :param center: bounding box in corner form (x,y) (x+w, y+h)
    :return: bounding box in center form (cx, cy, w, h)
    """
    return torch.cat([corner[..., 2:] / 2  + corner[..., :2] / 2,
                      corner[..., 2:] - corner[..., :2]], dim=-1)
