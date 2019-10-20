#! encoding: UTF-8
import os
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import json
import pickle
import time

from DualCNN import DualCNN
from gen import gen_all, gen_one
from opts import parse_arg
from train import forward


def train(args, model):
    state = {}
    log = {}
    if args.start_epoch != 0:
        model_path = os.path.join(args.model_dir, str(args.model_name) + str(args.start_epoch) + '.pth')
        assert os.path.exists(model_path), "model files at this epoch doesn't exist"
        state = torch.load(model_path)
        model.load_state_dict(state['model'])

        log_path = os.path.join(args.log_dir, str(args.model_name) + str(args.start_epoch) + '.pkl')
        if os.path.exists(log_path):
            log = torch.load(log_path)

    epoch = state.get('epoch', 0)

    # val_result_log = histories.get('val_result_log', {})
    loss_log = log.get('loss_log', {})

    for i in range(epoch, args.max_epoch):

        print (i + 1)
        t_start = time.time()
        train_loss, train_perp, train_sent_loss = forward(args, model, train=True)
        print (time.time() - t_start)
        print (
            "epoch {0} train_loss: {1:.2f}, train_perp: {2:.2f}, sent_loss: {3:.2f}".format(i + 1, train_loss,
                                                                                            train_perp,
                                                                                            train_sent_loss))
        test_loss, test_perp, test_sent_loss = forward(args, model, train=False)
        print (
            "epoch {0} test_loss: {1:.2f}, test_perp: {2:.2f}, sent_loss: {3:.2f}".format(i + 1, test_loss, test_perp,
                                                                                          test_sent_loss))
        state['model'] = model.state_dict()
        state['epoch'] = i + 1
        model_path = os.path.join(args.model_dir, str(args.model_name) + str(i + 1) + '.pth')
        torch.save(state, model_path)

        loss_log[i + 1] = {'test_loss': float(test_loss), 'test_perp': float(test_perp), \
                           'sent_loss': float(test_sent_loss)}
        log_path = os.path.join(args.log_dir, str(args.model_name) + str(i + 1) + '.pkl')
        torch.save(log, log_path)
        for img_id in args.gen_list:
            gen_one(args, model, img_id)
        if args.eval:
            gen_all(args, model, i + 1)


def eval_or_gen(args, model):
    for i in range(args.start_epoch, args.max_epoch):
        print (i)
        model_path = os.path.join(args.model_dir, str(args.model_name) + str(i) + '.pth')
        assert os.path.exists(model_path), "model files at this epoch doesn't exist"
        state = torch.load(model_path)
        model.load_state_dict(state['model'])
        if args.gen:
            for img_id in args.gen_list:
                gen_one(args, model, img_id)
        if args.eval:
            gen_all(args, model, i)


def main():
    args = parse_arg()
    args.device = None

    if torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = torch.device('cuda:' + str(args.cuda_id))
    else:
        args.device = torch.device('cpu')

    args.word2idx, args.idx2word = pickle.load(open(args.vocab_path, 'rb'))

    model = DualCNN(args).to(device=args.device)
    if args.train:
        train(args, model)
    else:
        eval_or_gen(args, model)


if __name__ == '__main__':
    main()












































    # def collate(data):
    #     data.sort(key=lambda p: p[3], reverse=True)
    #     img_feats, words, stop_flags, sents_num, words_num = zip(*data)
    #
    #     img_feats = np.asarray(img_feats)
    #     img_feats = torch.FloatTensor(img_feats)
    #
    #     words = torch.LongTensor(words)
    #     stop_flags = torch.LongTensor(stop_flags)
    #     words_num = torch.IntTensor(words_num)
    #     sents_num = np.asarray(sents_num)
    #
    #     return (img_feats, words, stop_flags, words_num, sents_num)



    # stop_flags = pack_padded_sequence(stop_flags, sents_num, batch_first=True)[0]
    # predict_stop = pack_padded_sequence(predict_stop, sents_num, batch_first=True)[0]
    # predict_words = pack_padded_sequence(predict_words, sents_num, batch_first=True)[0]
    # target_words = pack_padded_sequence(target_words, sents_num, batch_first=True)[0]
    # words_num = pack_padded_sequence(words_num, sents_num, batch_first=True)[0]
    # total_words_num = int(torch.sum(words_num))
    # _predict_words = torch.zeros(total_words_num, 4720)
    # _target_words = torch.LongTensor(total_words_num)
    # count = 0
    # for i, word_num in enumerate(words_num):
    #         _predict_words[count: count+word_num] = predict_words[i, :word_num, :]
    #         _target_words[count: count+word_num] = target_words[i, :word_num]
    #         count += word_num
    # _predict_words = Variable(_predict_words).cuda(args.gpu_id)
    # _target_words = Variable(_target_words).cuda(args.gpu_id)




    # stop_param = []
    # param_without_stop = []

    # for i, param in enumerate(model.parameters()):
    #     if i==14 or i==15:
    #         stop_param.append(param)
    #     else:
    #         param_without_stop.append(param)
    #
    # optimizer = optim.Adam([
    #     {'params': param_without_stop, 'weight_decay': 1e-5},
    #     {'params': stop_param, 'weight_decay': weigh_decay}
    #     ], lr=lr)

    # stop_mask = (stop_labels == 0)
    # stop_index = torch.sum(stop_mask, 1)
    # for i in range(real_batch_size):
    #     stop_mask[i, int(stop_index[i])] = 1
    # stop_mask = torch.FloatTensor(np.asarray(stop_mask))
