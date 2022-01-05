import os
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_imagelist(file_path, isTrain=False):
    imgs = os.listdir(file_path)
    img_list = [os.path.join(file_path, img) for img in imgs]
    return img_list

COCO_dir = '/w/247/zyanliu/COCO_data/'
train_imglist = get_imagelist(COCO_dir+'train2014', True)
val_imglist = get_imagelist(COCO_dir+'val2014', False)

import numpy as np
import pickle as pkl
def extract_save_img_feature(imgnames, save_path, classname):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature_dict = {}
    img_nums = len(imgnames)
    numpy_feat = None
    feat_dict = {}
    for i in range(0, img_nums, 8):
        if i % 200 == 0:
            print('batch idx', i)
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
            
        



save_basep = 'COCO_features_clip'
if not os.path.exists(save_basep):
    os.mkdir(save_basep)
# extract_save_img_feature(train_imgdict, os.path.join(save_basep, 'train'))
extract_save_img_feature(val_imglist, os.path.join(save_basep, 'val'), 'val')

extract_save_img_feature(train_imglist, os.path.join(save_basep, 'train'), 'train')
