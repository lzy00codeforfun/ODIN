import os
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_imagelist(file_path, isTrain=False):
    img_folders = os.listdir(file_path)
    img_dict = {}
    img_list = []
    for img_folder_t in img_folders:
        if isTrain:
            img_folder = os.path.join(file_path, img_folder_t, 'images')
        else:
            img_folder = os.path.join(file_path, img_folder_t)
        tmp_list = os.listdir(img_folder)
        img_dict[img_folder_t] = [os.path.join(img_folder, img) for img in tmp_list]
        img_list += [os.path.join(img_folder, img) for img in tmp_list]
    return img_dict, img_list

train_imgdict, train_imglist = get_imagelist('/w/247/zyanliu/ImageNet/tiny-imagenet-200/train', True)
# test_imgdict, test_imglist = get_imagelist('/w/247/zyanliu/ImageNet/tiny-imagenet-200/test')
val_imgdict, val_imglist = get_imagelist('/w/247/zyanliu/ImageNet/tiny-imagenet-200/val', False)

import numpy as np
import pickle as pkl
def extract_save_img_feature(img_dict, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature_dict = {}
    for classname, imgnames in img_dict.items():
        img_nums = len(imgnames)
        numpy_feat = None
        feat_dict = {}
        for i in range(0, img_nums, 8):
            tmp_imgnames = imgnames[i: i+min(8, img_nums-i)]
            # imgs = [preprocess(Image.open(imgnames[i+j])).unsqueeze(0) for j in range(0, min(8, img_nums-i), 1)]
            imgs = [preprocess(Image.open(timgname)).unsqueeze(0) for timgname in tmp_imgnames]
            imgs = torch.cat(imgs, axis=0).to(device)
            # print('imgs shape', imgs.shape)
            with torch.no_grad():
                image_features = model.encode_image(imgs).detach().cpu().numpy()
                if numpy_feat is None:
                    numpy_feat = image_features
                else:
                    numpy_feat = np.concatenate((numpy_feat, image_features), axis=0)
                for(img_feat, imgname) in zip(image_features, tmp_imgnames):
                    feat_dict[imgname] = img_feat
        with open(os.path.join(save_path, classname+'.pkl'), 'wb') as f:
            pkl.dump(feat_dict, f)
            
        

                # print('numpy shape', image_features.shape)


save_basep = 'features_clip'
if not os.path.exists(save_basep):
    os.mkdir(save_basep)
# extract_save_img_feature(train_imgdict, os.path.join(save_basep, 'train'))
extract_save_img_feature(val_imgdict, os.path.join(save_basep, 'val'))

