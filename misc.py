import ssd_net
import mobilenet
import bbox_helper
import torch
import torch.nn as nn
import torch.nn.functional as F
import module_util
import os
from collections import OrderedDict

torch.set_default_tensor_type('torch.cuda.FloatTensor')
#
current_directory = os.getcwd()  # current working directory
#
# model = ssd_net.SSD(num_classes=4)
# func = module_util.summary_layers(model, (3, 300, 300))

net = mobilenet.MobileNet()
net_state = net.state_dict()


pre_trained_model = os.path.join(current_directory, 'pretrained/mobienetv2.pth')

pret_state = torch.load(pre_trained_model)
print(pret_state.keys())
pret_state = {k: v for k, v in pret_state.items() if k in net.state_dict()}

net_state.update(pret_state)
net.load_state_dict(net_state)

print(net_state.values())


# print(type(pret_state))
# print(pret_state.keys())
# print(net.state_dict().keys())
# print(len(pret_state))
# print(len(net.state_dict()))
#net.load_state_dict(pret_state)

# prior_layer_cfg = [{'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (102, 102),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (144, 144),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (186, 186),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (228, 228),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
#                            {'layer_name': 'Conv17_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
#                             'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
#                            ]
#
# prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import module_util
# from mobilenet import MobileNet
# import os
# import numpy as np
#
# current_directory = os.getcwd()  # current working directory
#
#
# class SSD(nn.Module):
#     def __init__(self, num_classes):
#         super(SSD, self).__init__()
#         self.num_classes = num_classes
#
#         # Setup the backbone network (base_net)
#         self.base_net = MobileNet(num_classes)
#
#         # The feature map will extracted from layer [6] , layer[11] and layer[13] in (base_net)
#         self.base_output_layer_indices = (6, 11, 13)
#
#         # Define the Additional feature extractor
#         self.additional_feat_extractor = nn.ModuleList([
#             # Conv14_2
#             nn.Sequential(
#                 nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU()
#             ),
#             # Conv15_2
#             nn.Sequential(
#                 nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU()
#             ),
#             # Conv16_2
#             nn.Sequential(
#                 nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU()
#             ),
#             # Conv17_2
#             nn.Sequential(
#                 nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU()
#             )
#         ])
#
#         # Bounding box offset regressor
#         num_prior_bbox = 6  # num of prior bounding boxes
#         self.loc_regressor = nn.ModuleList([
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=1024, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=128, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1)
#             # TODO: implement remaining layers.
#         ])
#
#         # Bounding box classification confidence for each label
#         self.classifier = nn.ModuleList([
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=1024, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
#             nn.Conv2d(in_channels=128, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1)
#             # TODO: implement remaining layers.
#         ])
#
#         # Todo: load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
#         net_state = self.base_net.state_dict()
#         pre_trained_model = os.path.join(current_directory, 'pretrained/mobienetv2.pth')
#         pret_state = torch.load(pre_trained_model)
#         pret_state = {k: v for k, v in pret_state.items() if k in net_state}
#         net_state.update(pret_state)
#         self.base_net.load_state_dict(net_state)
#
#         # self.base_net.load_state_dict(pret_state)
#
#         def init_with_xavier(m):
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#
#         self.loc_regressor.apply(init_with_xavier)
#         self.classifier.apply(init_with_xavier)
#         self.additional_feat_extractor.apply(init_with_xavier)
#
#     def feature_to_bbbox(self, loc_regress_layer, confidence_layer, input_feature):
#         """
#         Compute the bounding box class scores and the bounding box offset
#         :param loc_regress_layer: offset regressor layer to run forward
#         :param confidence_layer: confidence layer to run forward
#         :param input_feature: feature map to be feed in
#         :return: confidence and location, with dim:(N, num_priors, num_classes) and dim:(N, num_priors, 4) respectively.
#         """
#         conf = confidence_layer(input_feature)
#         loc = loc_regress_layer(input_feature)
#
#         # Confidence post-processing:
#         # 1: (N, num_prior_bbox * n_classes, H, W) to (N, H*W*num_prior_bbox, n_classes) = (N, num_priors, num_classes)
#         #    where H*W*num_prior_bbox = num_priors
#         conf = conf.permute(0, 2, 3, 1).contiguous()
#         num_batch = conf.shape[0]
#         c_channels = int(conf.shape[1] * conf.shape[2] * conf.shape[3] / self.num_classes)
#         conf = conf.view(num_batch, c_channels, self.num_classes)
#
#         # Bounding Box loc and size post-processing
#         # 1: (N, num_prior_bbox*4, H, W) to (N, num_priors, 4)
#         loc = loc.permute(0, 2, 3, 1).contiguous()
#         loc = loc.view(num_batch, c_channels, 4)
#
#         return conf, loc
#
#     def forward(self, input):
#
#         confidence_list = []
#         loc_list = []
#
#         # Run the backbone network from [0 to 6, and fetch the bbox class confidence
#         # as well as position and size
#         y = module_util.forward_from(self.base_net.base_net, 0, self.base_output_layer_indices[0] + 1, input)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[0], self.classifier[0], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#         print("in SSD forward 1")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # Run the backbone network from [6 to 11, and fetch the bbox class confidence
#         # as well as position and size
#         y = module_util.forward_from(self.base_net.base_net, self.base_output_layer_indices[0] + 1,
#                                      self.base_output_layer_indices[1] + 1, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[1], self.classifier[1], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 2")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # Todo: implement run the backbone network from [11 to 13] and compute the corresponding bbox loc and confidence
#         # run from 11 to 13 in backbone and fetch confidence and loc_list
#         y = module_util.forward_from(self.base_net.base_net, self.base_output_layer_indices[1] + 1,
#                                      self.base_output_layer_indices[2] + 1, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[2], self.classifier[2], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 3")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # Todo: forward the 'y' to additional layers for extracting coarse features
#
#         # run from mobile net output to 1 additional_feat_extractor
#         y = module_util.forward_from(self.additional_feat_extractor, 0, 1, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[3], self.classifier[3], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 4")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # run from 1 to 2 additional_feat_extractor
#         y = module_util.forward_from(self.additional_feat_extractor, 1, 2, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[4], self.classifier[4], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 5")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # run from 2 to 3 additional_feat_extractor
#         y = module_util.forward_from(self.additional_feat_extractor, 2, 3, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[5], self.classifier[5], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 6")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         # run from 4 to last additional_feat_extractor
#         y = module_util.forward_from(self.additional_feat_extractor, 3, 4, y)
#         confidence, loc = self.feature_to_bbbox(self.loc_regressor[6], self.classifier[6], y)
#         confidence_list.append(confidence)
#         loc_list.append(loc)
#
#         print("in SSD forward 7")
#         print((torch.cat(confidence_list, 1)).shape)
#         print((torch.cat(loc_list, 1)).shape)
#         print(len(loc_list))
#
#         confidences = torch.cat(confidence_list, 1)
#         locations = torch.cat(loc_list, 1)
#
#         print(confidence.shape)
#         print(locations.shape)
#
#         # [Debug] check the output
#         assert confidence.dim() == 3  # should be (N, num_priors, num_classes)
#         assert locations.dim() == 3  # should be (N, num_priors, 4)
#         assert confidence.shape[2] == locations.shape[2]
#         assert locations.shape[2] == 4
#
#         if not self.training:
#             # If in testing/evaluating mode, normalize the output with Softmax
#             confidences = F.softmax(confidences, dim=2)
#
#         return confidences, locations
