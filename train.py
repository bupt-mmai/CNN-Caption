import torch
import torch.optim as optim
import math

from data_loader import img2para_dataset
from utils import cal_loss


def forward(args, model, train):
    if train:
        model.train()
    else:
        model.eval()

    dataset = img2para_dataset(args, train)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=train,
    )

    # rnn_params  = {'p':[], 's':[], 'w':[]}
    # for name, param in model.named_parameters():
    #     if 'pRNN' in name:
    #         rnn_params['p'].append(param)
    #     elif 'sRNN' in name:
    #         rnn_params['s'].append(param)
    #     elif 'wRNN' in name:
    #         rnn_params['w'].append(param)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_cost = 0
    total_word_cost = 0
    total_sent_cost = 0

    total_word_count = 0
    # total_sent_count = 0

    for batch_idx, batch_data in enumerate(data_loader):

        real_batch_size = len(batch_data[0])

        batch_data = [_.to(args.device) for _ in batch_data[1:]]
        img_feats, densecap, para_words_labels, stop_labels, words_mask, densecap_mask, fake_words, fake_words_mask = batch_data

        predict_words, predict_stop = model(img_feats, para_words_labels, words_mask, fake_words, fake_words_mask)

        para_words_count = torch.sum(words_mask)
        word_cost, sent_cost = cal_loss(para_words_labels, predict_words, words_mask, stop_labels, predict_stop)
        # para_sents_count = torch.sum(stop_mask)
        cost = (args.sent_cost_lambda * sent_cost + word_cost) / real_batch_size

        if train:
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        total_cost += cost.item()
        total_word_cost += word_cost.item()
        total_sent_cost += sent_cost.item() / real_batch_size
        total_word_count += para_words_count
        # total_sent_count += para_sents_count

        if train:
            print ("batch: {0} loss: {1:.2f}, perp: {2:.2f}, sent loss: {3:.2f}"
                   .format(batch_idx, word_cost.item() / real_batch_size, math.exp(word_cost.item() / para_words_count),
                           sent_cost.item() / real_batch_size))
    return total_cost / batch_idx, math.exp(total_word_cost / total_word_count), total_sent_cost / batch_idx
