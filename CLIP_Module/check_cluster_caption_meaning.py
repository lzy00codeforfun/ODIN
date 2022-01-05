fname = 'cluster_3_997.txt'
center_id_set = 3
import os
import shutil
def save_image_cluster(fname, center_id_set):
    f = open('cluster_res/'+fname, 'r').readlines()
    images = [i.split('\n')[0] for i in f]
    savePath = os.path.join('text_explanation/image_cluster/{}'.format(center_id_set))
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    for image in images:
        shutil.copy(image, savePath)

# save_image_cluster('cluster_3_997.txt', 3)
# save_image_cluster('cluster_27_1550.txt', 27)
# save_image_cluster('cluster_31_1359.txt', 31)

# cluster_res/cluster_41_675.txt cluster_res/cluster_22_274.txt cluster_res/cluster_5_591.txt
save_image_cluster('cluster_41_675.txt', 41)
save_image_cluster('cluster_22_274.txt', 22)
save_image_cluster('cluster_5_591.txt', 5)

exit()

f = open('cluster_res/'+fname, 'r').readlines()
images = [i.split('\n')[0] for i in f]
images = [i.split('/')[-1] for i in images]



from image_id_name_mapping import id_2_image_mapping_anno_train, image_2_id_mapping_anno_train
from image_id_name_mapping import id_2_image_mapping_anno_test, image_2_id_mapping_anno_test

import json

COCO_train_anno = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_train2014.json'))['annotations']
COCO_test_anno = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_val2014.json'))['annotations']

def get_textlist(coco_data):
    coco_dict = {}
    for data in coco_data:
        image_id = data['image_id']
        caption = data['caption']
        if image_id not in coco_dict:
            coco_dict[image_id] = []
        coco_dict[image_id].append(caption)
    return coco_dict

caption_dict_train = get_textlist(COCO_train_anno)
caption_dict_test = get_textlist(COCO_test_anno)

caption_textlist = []
for imgname in images:
    if 'train' in imgname:
        img_id = image_2_id_mapping_anno_train[imgname]
        caption_textlist += caption_dict_train[img_id]
    else:
        img_id = image_2_id_mapping_anno_test[imgname]
        caption_textlist += caption_dict_test[img_id]

frequency = {}
for text in caption_textlist:
    text = text.split(' ')
    for i in text:
        if i not in frequency:
            frequency[i] = 0
        frequency[i] += 1

# sorted_res = sorted(list(frequency.items()), key=lambda x: -x[1])
# print(caption_textlist)
f = open('text_explanation/{}_c.txt'.format(center_id_set), 'w')
f.write("\n".join(caption_textlist))
# print(sorted_res[:100])
