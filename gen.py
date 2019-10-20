import os
import json
import pickle
import torch
import numpy as np
import h5py

from data_loader import img2para_dataset


def gen_all(args, model, epoch):
    if epoch <= args.eval_after:
        return
    img2para = pickle.load(open(args.img2para_path, 'rb'))
    hypo = {}
    ref = {}

    dataset = img2para_dataset(args, False)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        # num_workers=args.num_workers,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    for batch_idx, batch_data in enumerate(data_loader):
        print (batch_idx)
        img_ids, img_feats = batch_data[0], batch_data[1].to(args.device)
        if args.beam:
            res = model.beam_search(img_feats, args.beam_size)
        else:
            res = model.sample(img_feats)
        for i, id in enumerate(img_ids):
            hypo[id] = [res[i]]
            ref_para = img2para[int(id)]
            tmp_para = ''

            for sent in ref_para:
                sent = sent.replace(',', ' , ')
                tmp_para += sent.lower() + ' . '
                # tmp_para += sent.lower() + ' '
            ref[id] = [tmp_para.strip()]
    result = (hypo, ref)
    type = 'beam' if args.beam else 'sample'
    result_f = open(
        os.path.join(args.eval_dir, str(args.model_name) + str(epoch) + type + str(args.beam_size) + '.txt'), 'w')
    json.dump(result, result_f)


def gen_one(args, model, img_id):
    test_img_names = json.load(open(args.test_img_path, 'r'))
    test_feats = h5py.File(args.test_feats_path, 'r').get('feats')
    img2dense = pickle.load(open(args.densecap_path, 'rb'))

    if isinstance(img_id, int):
        index = img_id
    elif isinstance(img_id, str):
        index = test_img_names.index(img_id)
    else:
        raise Exception('img_id TypeWrong')

    test_feat = test_feats[index]
    densecap = img2dense[test_img_names[index]]

    densecap_mask = np.zeros(shape=densecap.shape, dtype=np.float32)
    densecap_mask[densecap != 2] = 1

    tmp_data = [test_feat, densecap, densecap_mask]
    tmp_data = [torch.from_numpy(_).to(args.device) for _ in tmp_data]
    test_feat, densecap, densecap_mask = tmp_data

    if args.beam:
        print (model.beam_search(test_feat, args.beam_size))
    else:
        print (model.sample(test_feat))
