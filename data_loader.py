# -*- coding: utf-8 -*-
import os
import pickle
import torch.utils.data as data
import h5py
import json
import numpy as np
import random


class img2para_dataset(data.Dataset):
    def __init__(self, args, train):
        if train:
            imgs_path = args.train_img_path
            feats_path = args.train_feats_path
        else:
            imgs_path = args.test_img_path
            feats_path = args.test_feats_path
        self.imgs_name = json.load(open(imgs_path, 'r'))
        self.img_feats = h5py.File(feats_path, 'r').get('feats')
        self.img2para_vec = pickle.load(open(args.para2vec_path, 'rb'))
        self.densecap_vec = pickle.load(open(args.densecap_path, 'rb'))

    def __getitem__(self, index):
        img_name = str(self.imgs_name[index])

        img_feat = self.img_feats[index]

        para_words = self.img2para_vec[img_name][0]
        words_mask = np.zeros(shape=para_words.shape, dtype=np.float32)
        words_mask[para_words != 2] = 1

        fake_words = np.zeros(shape=para_words.shape, dtype=np.float32)
        for i in range(5):
            fake_img_idx = random.randint(0, len(self.imgs_name) - 1)
            if fake_img_idx == index:
                fake_img_idx += 1
            fake_img_name = self.imgs_name[fake_img_idx]
            random_sent_id = random.randint(0, 5)
            fake_words[i] = self.img2para_vec[fake_img_name][0][random_sent_id]
        fake_words_mask = np.zeros(shape=fake_words.shape, dtype=np.float32)
        fake_words_mask[fake_words != 2] = 1

        stop_labels = self.img2para_vec[img_name][1]
        densecap_vec = self.densecap_vec[img_name]
        densecap_mask = np.zeros(shape=densecap_vec.shape, dtype=np.float32)
        densecap_mask[densecap_vec != 2] = 1

        return img_name, img_feat, densecap_vec, para_words, stop_labels, words_mask, \
               densecap_mask, fake_words, fake_words_mask

    def __len__(self):
        return len(self.img_feats)
