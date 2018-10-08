import ssd_net
import mobilenet
import bbox_loss
import cityscape_dataset
import bbox_helper
from util import module_util
import os
from glob import glob
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pat

current_directory = os.getcwd() #current working directory

if __name__ == '__main__':
    polygons_label_path = "cityscapes_samples_labels"
    images_path = "cityscapes_samples"

    compl_poly_path = os.path.join(current_directory,polygons_label_path,"*","*_polygons.json")

    poly_folders = glob(compl_poly_path)

    poly_folders = np.array(poly_folders)

    print(poly_folders.shape)

    image_label_list =[]

    for file in poly_folders:
        with open(file,"r") as f:
            frame_info =json.load(f)
            length=len(frame_info['objects'])
            file_path = file
            image_name = file_path.split("/")[-1][:-23]
            for i in range(length):
                label = frame_info['objects'][i]['label']
                if label == "ego vehicle":
                    break
                polygon = np.array(frame_info['objects'][i]['polygon'],dtype=np.float32)
                left_top = np.min(polygon,axis=0)
                right_bottom =np.max(polygon,axis=0)
                ltrb= np.concatenate((left_top,right_bottom)).reshape(1,-1)
                image_label_list.append({'image_name':image_name,'file_path':file_path,'label':label,'bbox':ltrb})

    print(image_label_list)
    image_ll_len = len(image_label_list)
    print(image_ll_len)

    print(image_label_list[0]['file_path'])

    #get images list

    compl_img_path = os.path.join(current_directory,images_path,"*","*")

    images = glob(compl_img_path)

    images = np.array(images)

    print(images[0])

    for i in range(len(images)):
        print(images[i])
        print(images[i].split("/"))
        img_folder = images[i].split('/')[-2]
        img_name = images[i].split('/')[-1]
        print(img_folder)
        print(img_name[:-16])
        img_iden = img_name[:-16]
        image_path= os.path.join(current_directory,images_path,img_folder,img_name)
        image = Image.open(image_path)
        print(image.size)
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        for i in range(image_ll_len):
            if image_label_list[i]["image_name"] == img_iden:
                print(i)
                print(image_label_list[i]["label"])
                bbox = np.array(image_label_list[i]['bbox'], dtype=np.float32)
                print(bbox)
                print(bbox.shape)
                rect = pat.Rectangle((bbox[0][0],bbox[0][1]),bbox[0][2]-bbox[0][0],bbox[0][3]-bbox[0][1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)


        plt.show()

