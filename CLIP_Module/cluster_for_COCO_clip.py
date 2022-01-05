import os, time, json, sys, pdb
import argparse
import pickle as pkl
import numpy as np
import argparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
parser = argparse.ArgumentParser(description='cluster for clip')
parser.add_argument('-t', '--train_num', type=int, default=10, help='train label number')
parser.add_argument('-q', '--test_num', type=int, default=2, help='query label number')
parser.add_argument('-m', '--model', type=str, default='clip', help='model type')

args = parser.parse_args()

def transform_pkl2npy(pkl_path):
    pkl_dict = pkl.load(open(pkl_path, 'rb'))
    imgfeat_list = []
    imgname_list = []
    for imgname, feat_npy in pkl_dict.items():
        imgfeat_list.append(feat_npy)
        imgname_list.append(imgname)

    # print(npy_img[0].shape)
    # npy_img = np.concatenate(npy_img, axis=0)
    
    return imgfeat_list, imgname_list

train_imgfeats, train_imgnames = transform_pkl2npy('/w/247/zyanliu/CLIP/COCO_features_clip/train/train.pkl')
test_imgfeats, test_imgnames = transform_pkl2npy('/w/247/zyanliu/CLIP/COCO_features_clip/val/val.pkl')

tot_imgfeats = train_imgfeats + test_imgfeats
tot_imgnames = train_imgnames + test_imgnames
print('tot imgfeats', len(tot_imgfeats), tot_imgfeats[0].shape)
import time

def scatter_plot(latent_representations, labels):
    '''
    the scatter plot for visualizing the latent representations with the ground truth class label
    ----------
    latent_presentations: (N, dimension_latent_representation)
    labels: (N, )  the labels of the ground truth classes
    '''
    # borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if the dimension_latent_representation > 2 you need to apply TSNE transformation
    # to map the latent representations from higher dimensionality to 2D
    # You can use #from sklearn.manifold import TSNE#

    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], cmap=discrete_cmap(10, 'jet'), c=labels, edgecolors='black')
    plt.colorbar()
    plt.grid()
    plt.show()
    plt.savefig('fig_folder/clip_coco_cluster.png', format='png')

from sklearn import manifold
def tsne_trans(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    x_sne_res = tsne.fit_transform(X)
    return x_sne_res

# from get_tsne_plot_imagenet import tsne_trans, scatter_plot

def main():
    st = time.time()
    k = 30
    print('start kmeans')
    minik = MiniBatchKMeans(n_clusters=k, batch_size = 150, random_state=9)
    kmeans = KMeans(n_clusters=k, random_state=9)
    y_pred = minik.fit_predict(tot_imgfeats)
    # y_pred = kmeans.fit_predict(tot_imgfeats)
    print('predict done', time.time()-st)
    print(y_pred[:100])

    embed_tsne = tsne_trans(tot_imgfeats)
    print(len(embed_tsne), len(embed_tsne[0]), type(embed_tsne), type(embed_tsne[0]))
    scatter_plot(embed_tsne, tot_imgnames)


if __name__ == "__main__":
    main()
