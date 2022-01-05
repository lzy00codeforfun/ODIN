import json

COCO_train_anno = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_train2014.json'))['images']
COCO_test_anno = json.load(open('/w/247/zyanliu/COCO_data/annotations/captions_val2014.json'))['images']

COCO_train_in = json.load(open('/w/247/zyanliu/COCO_data/annotations/instances_train2014.json'))['images']
COCO_test_in = json.load(open('/w/247/zyanliu/COCO_data/annotations/instances_val2014.json'))['images']

image_2_id_mapping_in_train = {}
image_2_id_mapping_in_test = {}
image_2_id_mapping_anno_train = {}
image_2_id_mapping_anno_test = {}
for img in COCO_train_in:
    image_2_id_mapping_in_train[img['file_name']] = img['id']
for img in COCO_test_in:
    image_2_id_mapping_in_test[img['file_name']] = img['id']

for img in COCO_train_anno:
    image_2_id_mapping_anno_train[img['file_name']] = img['id']
for img in COCO_test_anno:
    image_2_id_mapping_anno_test[img['file_name']] = img['id']

def get_rev(d):
    res= {}
    for i,j in d.items():
        res[j] = i
    return res

id_2_image_mapping_in_train = get_rev(image_2_id_mapping_in_train)
id_2_image_mapping_in_test = get_rev(image_2_id_mapping_in_test)
id_2_image_mapping_anno_train = get_rev(image_2_id_mapping_anno_train)
id_2_image_mapping_anno_test = get_rev(image_2_id_mapping_anno_test)

print('get mapping done')
