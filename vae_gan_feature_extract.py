import os, shutil, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary, torchvision
import torchvision.transforms as  T
from PIL import Image


def main(config, mode, weights):
    # Generate configuration
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)
    print("config :", config)

    # Generate logger
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


    """ Load previousely saved logger, if it exists """
    DRIVE_BACKUP = config.get("SAVE.DRIVE_BACKUP")
    if DRIVE_BACKUP:
        backup_logger = os.path.join(CHECKPOINT_DIRECTORY, LOGGER_SAVE_NAME)
        if os.path.exists(backup_logger):
            shutil.copy2(backup_logger, ".")
    else:
        backup_logger = None

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs"%NUM_GPUS)


    # --------------------- BUILD GENERATORS ------------------------
    # Supported integrated data sources --> MNIST, CIFAR
    # For BDD or others need a crawler and stuff...but we;ll deal with it later
    from generators import ClassedGenerator_tinyImageNet as ClassedGenerator

    load_dataset = config.get("EXECUTION.DATASET_PRELOAD")
    if load_dataset in ["MNIST", "CIFAR10", "CIFAR100", "tinyImageNet"]:
        crawler = load_dataset
        #dataset = torchvision.datasets.MNIST(root="./MNIST", train=True,)
        logger.info("No crawler necessary when using %s dataset"%crawler)
    else:
        raise NotImplementedError()

    
    test_generator = ClassedGenerator( gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                        normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                        h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                        **TRAINDATA_KWARGS)    
    test_generator.setup(  crawler, preload_classes = config.get("EXECUTION.DATASET_TEST_PRELOAD_CLASS"), \
                            mode='val',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), \
                            workers = config.get("TRANSFORMATION.WORKERS"))
    logger.info("Generated testing data generator")


    # --------------------- INSTANTIATE MODEL ------------------------
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("EXECUTION.MODEL_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.MODEL_BUILDER"), "models"))

    vaegan_model = model_builder(   arch=config.get("MODEL.ARCH"), base=config.get("MODEL.BASE"), \
                                    latent_dimensions = config.get("MODEL.LATENT_DIMENSIONS"), \
                                    **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")

    if mode == "test":
        msg = vaegan_model.load_state_dict(torch.load(weights))
        logger.info("load state dict results")
        logger.info(msg)
        vaegan_model.cuda()
        vaegan_model.eval()

    
    if not os.path.exists('feature_vae'):
        os.mkdir('feature_vae')
    transforms_test = get_transforms(i_shape = config.get("DATASET.SHAPE"), \
            t_crop = config.get("TRANSFORMATION.T_CROP"))
    vaegan_model.cuda()
    extract_save_img_feature(vaegan_model.Encoder, train_imgdict, 'feature_vae/train', transforms_test)
    extract_save_img_feature(vaegan_model.Encoder, val_imgdict, 'feature_vae/val', transforms_test)
    

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
val_imgdict, val_imglist = get_imagelist('/w/247/zyanliu/ImageNet/tiny-imagenet-200/val', False)

def get_transforms(i_shape, t_crop):
    transformer_primitive = []
    transformer_primitive.append(T.Resize(size=i_shape))
    if t_crop:
        transformer_primitive.append(T.CenterCrop(size=i_shape))
    transformer_primitive.append(T.ToTensor())
    # transformer_primitive.append(T.RandomErasing(p=0.5, scale=(0.02, 0.4), value = kwargs.get('rea_value', 0)))
    return T.Compose(transformer_primitive)

import numpy as np
import pickle as pkl
def extract_save_img_feature(model, img_dict, save_path, transforms_test):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature_dict = {}
    zeros = torch.zeros(1,3, 64, 64)
    print(save_path, len(list(img_dict.items())))
    for classname, imgnames in img_dict.items():
        img_nums = len(imgnames)
        numpy_feat = None
        feat_dict = {}
        for i in range(0, img_nums, 8):
            tmp_imgnames = imgnames[i: i+min(8, img_nums-i)]
            # imgs = [preprocess(Image.open(imgnames[i+j])).unsqueeze(0) for j in range(0, min(8, img_nums-i), 1)]
            # imgs = [preprocess(Image.open(timgname)).unsqueeze(0) for timgname in tmp_imgnames]
            imgs = [transforms_test(Image.open(timgname)).unsqueeze(0) for timgname in tmp_imgnames]
            imgs = [t_img if t_img.shape == (1,3, 64,64) else zeros for t_img in imgs ]
            # print(imgs[0].shape, imgs[1].shape)
            try:
                imgs = torch.cat(imgs, axis=0).cuda()
            except:
                for img in imgs:
                    print(img.shape)
                assert(1==2)
            # print('imgs shape', imgs.shape)
            with torch.no_grad():
                image_features = model(imgs).detach().cpu().numpy()
                if numpy_feat is None:
                    numpy_feat = image_features
                else:
                    numpy_feat = np.concatenate((numpy_feat, image_features), axis=0)
                    for(img_feat, imgname) in zip(image_features, tmp_imgnames):
                        feat_dict[imgname] = img_feat
        with open(os.path.join(save_path, classname+'.pkl'), 'wb') as f:
            pkl.dump(feat_dict, f)


if __name__ == "__main__":
    ckpt_path = "tinyImageNet_v1_final_120ep_model/tinyImageNet_v1-v1_epoch115.pth"
    # weights = torch.load_state_dict(ckpt_path)
    main("config/vaegan_tinyImageNet.yml", "test", ckpt_path)
    # main()
