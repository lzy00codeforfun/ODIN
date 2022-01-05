import os, time, json, sys, pdb
import argparse
import pickle as pkl
import numpy as np
import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets

import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import mlep.data_encoder.w2vGoogleNews as w2vGoogleNews
import mlep.trackers.MemoryTracker as MemoryTracker

import mlep.drift_detector.UnlabeledDriftDetector.KullbackLeibler as KullbackLeibler
# import mlep.text.DataCharacteristics.CosineSimilarityDataCharacteristics as CosineSimilarityDataCharacteristics

# import mlep.tools.distributions.CosineSimilarityDistribution as CosineSimilarityDistribution
# from mlep.tools.distributions.CosineSimilarityDistribution import CosineSimilarityDistribution
from mlep.tools.distributions.DistanceDistribution import DistanceDistribution as CosineSimilarityDistribution
import mlep.tools.metrics.TextMetrics as TextMetrics
import mlep.tools.metrics.NumericMetrics as NumericMetrics
import mlep.utils.array_utils as array_utils

import warnings
# warnings.filterwarnings(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt

from alibi_detect.cd import MMDDrift

import traceback
import collections
parser = argparse.ArgumentParser(description='input one integer')
parser.add_argument('--train_label_num', '-t', type=int, help='train label num')
parser.add_argument('--query_label_num', '-q', type=int, help='query label num')
parser.add_argument('--query_num', '-n', type=int, help='query image num per label')
parser.add_argument('--model', '-m', type=str, help='model to be used')

args = parser.parse_args()
print('----------------')
print(args.train_label_num, args.query_label_num, args.query_num, args.model)

# from torchvision.datasets import ImageFolder
# tot_label = ImageFolder(root="/w/247/zyanliu/ImageNet/tiny-imagenet-200/train").class_to_idx
tot_label_dict = {'n01443537': 0, 'n01629819': 1, 'n01641577': 2, 'n01644900': 3, 'n01698640': 4, 'n01742172': 5, 'n01768244': 6, 'n01770393': 7, 'n01774384': 8, 'n01774750': 9, 'n01784675': 10, 'n01855672': 11, 'n01882714': 12, 'n01910747': 13, 'n01917289': 14, 'n01944390': 15, 'n01945685': 16, 'n01950731': 17, 'n01983481': 18, 'n01984695': 19, 'n02002724': 20, 'n02056570': 21, 'n02058221': 22, 'n02074367': 23, 'n02085620': 24, 'n02094433': 25, 'n02099601': 26, 'n02099712': 27, 'n02106662': 28, 'n02113799': 29, 'n02123045': 30, 'n02123394': 31, 'n02124075': 32, 'n02125311': 33, 'n02129165': 34, 'n02132136': 35, 'n02165456': 36, 'n02190166': 37, 'n02206856': 38, 'n02226429': 39, 'n02231487': 40, 'n02233338': 41, 'n02236044': 42, 'n02268443': 43, 'n02279972': 44, 'n02281406': 45, 'n02321529': 46, 'n02364673': 47, 'n02395406': 48, 'n02403003': 49, 'n02410509': 50, 'n02415577': 51, 'n02423022': 52, 'n02437312': 53, 'n02480495': 54, 'n02481823': 55, 'n02486410': 56, 'n02504458': 57, 'n02509815': 58, 'n02666196': 59, 'n02669723': 60, 'n02699494': 61, 'n02730930': 62, 'n02769748': 63, 'n02788148': 64, 'n02791270': 65, 'n02793495': 66, 'n02795169': 67, 'n02802426': 68, 'n02808440': 69, 'n02814533': 70, 'n02814860': 71, 'n02815834': 72, 'n02823428': 73, 'n02837789': 74, 'n02841315': 75, 'n02843684': 76, 'n02883205': 77, 'n02892201': 78, 'n02906734': 79, 'n02909870': 80, 'n02917067': 81, 'n02927161': 82, 'n02948072': 83, 'n02950826': 84, 'n02963159': 85, 'n02977058': 86, 'n02988304': 87, 'n02999410': 88, 'n03014705': 89, 'n03026506': 90, 'n03042490': 91, 'n03085013': 92, 'n03089624': 93, 'n03100240': 94, 'n03126707': 95, 'n03160309': 96, 'n03179701': 97, 'n03201208': 98, 'n03250847': 99, 'n03255030': 100, 'n03355925': 101, 'n03388043': 102, 'n03393912': 103, 'n03400231': 104, 'n03404251': 105, 'n03424325': 106, 'n03444034': 107, 'n03447447': 108, 'n03544143': 109, 'n03584254': 110, 'n03599486': 111, 'n03617480': 112, 'n03637318': 113, 'n03649909': 114, 'n03662601': 115, 'n03670208': 116, 'n03706229': 117, 'n03733131': 118, 'n03763968': 119, 'n03770439': 120, 'n03796401': 121, 'n03804744': 122, 'n03814639': 123, 'n03837869': 124, 'n03838899': 125, 'n03854065': 126, 'n03891332': 127, 'n03902125': 128, 'n03930313': 129, 'n03937543': 130, 'n03970156': 131, 'n03976657': 132, 'n03977966': 133, 'n03980874': 134, 'n03983396': 135, 'n03992509': 136, 'n04008634': 137, 'n04023962': 138, 'n04067472': 139, 'n04070727': 140, 'n04074963': 141, 'n04099969': 142, 'n04118538': 143, 'n04133789': 144, 'n04146614': 145, 'n04149813': 146, 'n04179913': 147, 'n04251144': 148, 'n04254777': 149, 'n04259630': 150, 'n04265275': 151, 'n04275548': 152, 'n04285008': 153, 'n04311004': 154, 'n04328186': 155, 'n04356056': 156, 'n04366367': 157, 'n04371430': 158, 'n04376876': 159, 'n04398044': 160, 'n04399382': 161, 'n04417672': 162, 'n04456115': 163, 'n04465501': 164, 'n04486054': 165, 'n04487081': 166, 'n04501370': 167, 'n04507155': 168, 'n04532106': 169, 'n04532670': 170, 'n04540053': 171, 'n04560804': 172, 'n04562935': 173, 'n04596742': 174, 'n04597913': 175, 'n06596364': 176, 'n07579787': 177, 'n07583066': 178, 'n07614500': 179, 'n07615774': 180, 'n07695742': 181, 'n07711569': 182, 'n07715103': 183, 'n07720875': 184, 'n07734744': 185, 'n07747607': 186, 'n07749582': 187, 'n07753592': 188, 'n07768694': 189, 'n07871810': 190, 'n07873807': 191, 'n07875152': 192, 'n07920052': 193, 'n09193705': 194, 'n09246464': 195, 'n09256479': 196, 'n09332890': 197, 'n09428293': 198, 'n12267677': 199}
rev_dict = {}
for i,j in tot_label_dict.items():
    rev_dict[j] = i
tot_label_dict = rev_dict

train_label_list = []
test_label_list = []

train_num = args.train_label_num
test_num = args.query_label_num

for idx in range(0,train_num):
    train_label_list.append(tot_label_dict[idx])
for idx in range(120, 120+test_num):
    test_label_list.append(tot_label_dict[idx])

train_embeddings = []

query_inline_embeddings = []
query_outline_embeddings = []

from mlep.representations.ZonedDistribution import ZonedDistribution
from sklearn.metrics.pairwise import cosine_similarity

def transform_pkl2npy(pkl_dict):
    npy_img = []
    for imgname in list(pkl_dict.keys()):
        npy_img.append(np.expand_dims(pkl_dict[imgname], axis=0))
    # print(npy_img[0].shape)
    npy_img = np.concatenate(npy_img, axis=0)
    
    return npy_img, npy_img.mean(axis=0)

def getDistance(queryPoint, centroid):
        """ Compute cosine similarity

        The result is in [0,1]. After inversion (1-*), a value closer to 0 means similar, while a value closer to 1 means dissimilar.

        """
        return np.linalg.norm(queryPoint.reshape(-1) - centroid.reshape(-1))
        return 1.0 - cosine_similarity(queryPoint.reshape(1,-1), centroid.reshape(1,-1)).mean()
if args.model == 'vae':
    querybasepath = '/w/247/zyanliu/ODIN/feature_vae/val'
    trainbasepath = '/w/247/zyanliu/ODIN/feature_vae/train' 
elif args.model=='clip':
    querybasepath = '/w/247/zyanliu/CLIP/features_clip/val'
    trainbasepath = '/w/247/zyanliu/CLIP/features_clip/train'
else:
    querybasepath = '/w/247/zyanliu/ODIN/feature_mocov2/val'
    trainbasepath = '/w/247/zyanliu/ODIN/feature_mocov2/train' 

# feature_mocov2
def make_prediction(cd, query_list, isFlag):
    tp, tn, fp, fn = 0, 0, 0, 0
    # preds = cd.predict(query_list, drift_type='batch', return_p_val=True, return_distance=True)
    # print(preds['data']['is_drift'])
    # print(preds['data']['p_val'])
    # print(len(preds['data']['p_val']))
    # print('done predict batch')
    # return False
    for x in query_list:
        # print(x.shape, type(x))
        # preds = cd.predict(x)
        preds = cd.predict(np.expand_dims(x, axis=0), return_p_val=True, return_distance=True)
        if preds['data']['is_drift']:
            if isFlag:
                fn += 1
            else:
                tn += 1
        else:
            if isFlag:
                tp += 1
            else:
                fp += 1
    print('predict done')
    return tp, tn, fp, fn

def main():
    pos_cnt = 0
    neg_cnt = 0
    # update as per experiment requires
    # Checking Kullback Leibler...
    
    print('start main')    

    centroid_list = []
    train_embeddings_list = []
    for lidx, train_label in enumerate(train_label_list):
        # load from path
        f_path = os.path.join(trainbasepath, train_label+'.pkl')
        train_embeddings_tmp = pkl.load(open(f_path, 'rb'))

        # get embeddings and centroid
        train_embeddings_tmp, centroid = transform_pkl2npy(train_embeddings_tmp)
        # print(train_embeddings_tmp.shape, centroid.shape)
        centroid_list.append(np.expand_dims(centroid, axis=0))
        train_embeddings_list.append(train_embeddings_tmp)
        # get distribution by building from zoned distribution at once
    
    train_embeddings_tmp = np.concatenate(train_embeddings_list, axis=0)
    centroid = np.concatenate(centroid_list, axis=0).mean(axis=0)
    print('train embeddings shape' ,train_embeddings_tmp.shape)
    print('centroid shape', centroid.shape)
    
    p_val = .05
    drift_detector = MMDDrift(train_embeddings_tmp, p_val=p_val)
    
    # query for results
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for lidx, train_label in enumerate(train_label_list):
        # print('query for ', train_label)
        f_path = os.path.join(querybasepath, train_label+'.pkl')
        train_embeddings_tmp = pkl.load(open(f_path, 'rb'))
        train_embeddings_tmp, _ = transform_pkl2npy(train_embeddings_tmp)
        pos_cnt += train_embeddings_tmp.shape[0]

        tp_tmp, tn_tmp, fp_tmp, fn_tmp = make_prediction(drift_detector, train_embeddings_tmp, True)
        tp += tp_tmp
        tn += tn_tmp
        fp += fp_tmp 
        fn += fn_tmp

    for lidx, test_label in enumerate(test_label_list):
        # nprint('query for ', test_label)
        f_path = os.path.join(querybasepath, test_label+'.pkl')
        train_embeddings_tmp = pkl.load(open(f_path, 'rb'))
        train_embeddings_tmp, _ = transform_pkl2npy(train_embeddings_tmp)
        train_embeddings_tmp = train_embeddings_tmp[:args.query_num]
        neg_cnt += train_embeddings_tmp.shape[0]
        
        tp_tmp, tn_tmp, fp_tmp, fn_tmp = make_prediction(drift_detector, train_embeddings_tmp, False)
        tp += tp_tmp
        tn += tn_tmp
        fp += fp_tmp 
        fn += fn_tmp


    print('tp', tp, 'tn', tn, 'fp', fp, 'fn', fn, 'accuracy', (tp+tn)/(tp+tn+fp+fn))
    print('pos_cnt', pos_cnt, 'neg_cnt', neg_cnt, 'outlier rate', neg_cnt/(neg_cnt+pos_cnt))
    exit()
        

    
if __name__ == "__main__":
    main()
