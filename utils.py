import torch
import torch.nn.functional as F


def cal_loss(words_labels, predict_words, words_mask, stop_labels, predict_stop):
    words_labels = words_labels[:, :, 1:].contiguous().view(-1, 1)
    words_mask = words_mask[:, :, 1:].contiguous().view(-1, 1)
    predict_words = F.log_softmax(predict_words, -1).view(-1, predict_words.size(3))
    word_cost = - torch.gather(predict_words, 1, words_labels) * words_mask
    word_cost = torch.sum(word_cost)

    predict_stop = F.log_softmax(predict_stop, -1).view(-1, 2)
    stop_labels = stop_labels.view(-1, 1)
    sent_cost = -torch.gather(predict_stop, 1, stop_labels)
    sent_cost = torch.sum(sent_cost)

    return word_cost, sent_cost
