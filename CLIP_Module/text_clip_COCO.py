import json
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

COCO_train = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_train2014.json'))['annotations']
COCO_test = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_val2014.json'))['annotations']

from image_id_name_mapping import image_2_id_mapping_anno_train, id_2_image_mapping_anno_train, id_2_image_mapping_anno_test

import os
import pickle as pkl
def get_image_feat(coco_data, mode, savePath):
    if os.path.exists(savePath):
        with open(savePath, 'rb') as f:
            final_res = pkl.load(f)
            imgname_list = final_res['imgname_list']
            feat_list = final_res['feat_list']
        return imgname_list, feat_list

    text_embed_dict = {}
    for idx, data in enumerate(coco_data):
        if idx % 1000 == 0:
            print('batch idx', idx)
            # break
        text = data["caption"]
        if mode == 'train':
            imgname = id_2_image_mapping_anno_train[data["image_id"]]
        else:
            imgname = id_2_image_mapping_anno_test[data['image_id']]
        if imgname not in text_embed_dict:
            text_embed_dict[imgname] = []
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_embed_dict[imgname].append(text_features)
    final_res = {}
    imgname_list = []
    feat_list = []
    for imgname, text_feat in text_embed_dict.items():
        if len(text_feat) > 1:
            text_feat = torch.cat(text_feat, dim=0).mean(dim=0)
        else:
            text_feat = text_feat[0]

        text_feat = text_feat.detach().cpu().numpy()
        final_res[imgname] = text_feat
        imgname_list.append(imgname)
        feat_list.append(text_feat)
   
    final_res = {}
    final_res['imgname_list'] = imgname_list
    final_res['feat_list'] = feat_list
    with open(savePath, 'wb') as f:
        pkl.dump(final_res, f)
    return imgname_list, feat_list

if not os.path.exists('text_feat_folder'):
    os.mkdir('text_feat_folder')
savePath = 'text_feat_folder/'
# train_imglist, train_featlist = get_image_feat(COCO_train, 'train', savePath+'feat_train.pkl')
test_imglist, test_featlist = get_image_feat(COCO_test, 'test', savePath+'feat_test.pkl')

tot_imglist = train_imglist + test_imglist
tot_featlist = train_featlist + test_featlist



            
