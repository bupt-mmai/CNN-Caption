# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def Conv1d(in_channels, out_channels, kernel_size, padding):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    # Xavier Init
    std = math.sqrt((4 / (kernel_size * in_channels)))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    # return m with weight normalization
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, word2idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)

    # Embedding of <padding> is set to 0
    m.weight.data[padding_idx].fill_(0)
    return m


def Linear(in_features, out_features, activation=None):
    m = nn.Linear(in_features, out_features)
    # Xavier Init
    m.weight.data.normal_(mean=0, std=math.sqrt((2 / (in_features + out_features))))
    m.bias.data.zero_()
    # return m with weight normalization
    return nn.utils.weight_norm(m)


class DualCNN(nn.Module):
    def __init__(self, args):
        super(DualCNN, self).__init__()

        self.emb_size = args.emb_size
        self.s_kernel_size = args.s_kernel_size
        self.s_padding = self.s_kernel_size - 1
        self.w_kernel_size = args.w_kernel_size
        self.w_padding = self.w_kernel_size - 1
        self.s_num_layers = args.s_num_layers
        self.w_num_layers = args.w_num_layers
        self.s_max = args.s_max
        self.w_max = args.w_max
        self.feat_size = args.feat_size
        self.proj_size = args.proj_size
        self.num_boxes = args.num_boxes
        self.idx2word = args.idx2word
        self.word2idx = args.word2idx
        self.vocab_len = len(self.idx2word)
        self.padding_idx = args.pad_idx

        self.softmax = nn.Softmax(-1)
        self.project = nn.Sequential(
            Linear(self.feat_size, self.proj_size),
            nn.ReLU()
        )
        self.sCNN = nn.ModuleList()
        self.wCNN = nn.ModuleList()
        self.device = args.device

        for i in range(self.s_num_layers):
            i_size = self.proj_size + self.emb_size if i == 0 else self.emb_size
            o_size = 2 * self.emb_size
            self.sCNN.append(Conv1d(i_size, o_size, self.s_kernel_size, self.s_padding))

        for i in range(self.w_num_layers):
            self.wCNN.append(Conv1d(self.emb_size, 2 * self.emb_size, self.w_kernel_size, self.w_padding))

        self.gen_topic = nn.Sequential(
            Linear(self.emb_size, self.emb_size, 'relu'),
            nn.ReLU(),
            Linear(self.emb_size, self.emb_size, 'relu'),
            nn.ReLU()
        )

        self.w_res_proj = Linear(2 * self.emb_size, self.emb_size)
        self.s_res_proj = Linear(self.emb_size + self.proj_size, self.emb_size)

        # self.sent_proj = nn.Sequential(
        #     Linear(self.emb_size, self.emb_size, 'relu'),
        #     nn.ReLU()
        # )

        self.stop_classifier = Linear(self.emb_size, 2)
        self.classifier = Linear(self.emb_size, self.vocab_len)
        self.embedding = Embedding(self.vocab_len, self.emb_size, self.padding_idx, self.word2idx)

        # self.ln2 = nn.LayerNorm([self.s_max, self.emb_size])
        # self.ln3 = nn.LayerNorm([self.emb_size])
        # self.ln4 = nn.LayerNorm([self.s_max, self.emb_size])

    def forward(self, img_feats, para_words, words_mask, fake_words, fake_words_mask):

        real_bsz = para_words.size(0)

        proj_feats = self.project(img_feats)
        pool_feats = torch.max(proj_feats, 1)[0].unsqueeze(1) / math.sqrt(2)

        label_words_embed = self.embedding(para_words)

        # ignore these, just for try
        # fake_words_embed = self.embedding(fake_words)[:, :-1, 1:, :]
        # fake_sents = torch.max(fake_words_embed, 2)[0]

        pred_words = label_words_embed.new_zeros(real_bsz, self.s_max, self.w_max, self.vocab_len)

        # <start> token embedding
        start_words = torch.zeros((real_bsz, 1), dtype=torch.long, device=self.device)
        start_words_embed = self.embedding(start_words)

        sents_embed = torch.max(label_words_embed[:, :-1, 1:, :], 2)[0]
        sents_embed = torch.cat([start_words_embed, sents_embed], 1) * math.sqrt(2)
        # sents_embed = self.ln2(sents_embed)

        s_inputs = pool_feats.expand(real_bsz, self.s_max, self.proj_size)
        s_inputs = torch.cat([s_inputs, s_inputs.new_zeros(real_bsz, self.s_max, self.emb_size)], 2)
        s_inputs[:, :, self.proj_size:] = sents_embed

        s_inputs = s_inputs.transpose(2, 1)
        for i, conv in enumerate(self.sCNN):
            res = s_inputs if i != 0 else self.s_res_proj(s_inputs.transpose(2, 1)).transpose(2, 1)

            s_outputs = conv(s_inputs)
            s_outputs = s_outputs[:, :, :-self.s_padding]
            s_outputs = F.glu(s_outputs, dim=1)
            s_outputs = self.ln3(s_outputs.transpose(2, 1)).transpose(2, 1)

            att_res = s_outputs
            att_weight = torch.bmm(proj_feats, s_outputs)
            att_weight = att_weight / math.sqrt(self.emb_size)
            att_weight = nn.Softmax(1)(att_weight)
            s_outputs = torch.bmm(proj_feats.permute(0, 2, 1), att_weight)

            s_outputs = (att_res + s_outputs) * math.sqrt(.5)
            s_outputs = (s_outputs + res) * math.sqrt(.5)
            s_inputs = s_outputs

        s_outputs = s_outputs.transpose(2, 1)
        sent_topics = self.gen_topic(s_outputs)
        pred_stop = self.stop_classifier(sent_topics)

        # wCNN
        for sent_id in range(self.s_max):
            topic = sent_topics[:, sent_id:sent_id + 1, :]
            words = label_words_embed[:, sent_id, :-1, :]
            w_inputs = torch.cat([topic, words], 1).transpose(2, 1)
            for i, conv in enumerate(self.wCNN):
                res = w_inputs
                w_outputs = conv(w_inputs)
                w_outputs = w_outputs[:, :, :-self.w_padding]
                w_outputs = F.glu(w_outputs, dim=1)
                w_outputs = (w_outputs + res) * math.sqrt(.5)
                w_inputs = w_outputs
            w_outputs = w_outputs.transpose(2, 1)
            pred_words[:, sent_id, :, :] = self.classifier(w_outputs[:, 1:, :])

        return pred_words, pred_stop

    def beam_search(self, img_feats, beam_size=2):

        img_feats = img_feats.view(-1, self.num_boxes, img_feats.size(-1))

        real_bts = img_feats.size(0)

        proj_feats = self.project(img_feats)
        pool_feats = torch.max(proj_feats, 1)[0].unsqueeze(1) / math.sqrt(2)

        result = ['' for _ in range(real_bts)]

        global_s_inputs = pool_feats.expand(real_bts, self.s_max, self.proj_size)
        global_s_inputs = torch.cat([global_s_inputs, global_s_inputs.new_zeros(real_bts, self.s_max, self.emb_size)],
                                    2)
        global_s_inputs = global_s_inputs.transpose(2, 1)

        start_words = torch.zeros((real_bts, 1), dtype=torch.long, device=self.device)
        start_words = self.embedding(start_words)

        for sent_id in range(self.s_max):

            if sent_id == 0:
                prev_sents_embed = start_words.squeeze(1)

            global_s_inputs[:, self.proj_size:, sent_id] = prev_sents_embed
            s_inputs = global_s_inputs.clone()
            for i, conv in enumerate(self.sCNN):
                res = s_inputs if i != 0 else self.s_res_proj(s_inputs.transpose(2, 1)).transpose(2, 1)
                s_outputs = conv(s_inputs)
                s_outputs = s_outputs[:, :, :-self.s_padding]
                s_outputs = F.glu(s_outputs, dim=1)
                s_outputs = self.ln3(s_outputs.transpose(2, 1)).transpose(2, 1)
                att_res = s_outputs

                att_weight = torch.bmm(proj_feats, s_outputs)
                att_weight = att_weight / math.sqrt(self.emb_size)
                att_weight = nn.Softmax(1)(att_weight)
                s_outputs = torch.bmm(proj_feats.permute(0, 2, 1), att_weight)

                s_outputs = (s_outputs + att_res) * math.sqrt(.5)
                s_outputs = (s_outputs + res) * math.sqrt(.5)

                s_inputs = s_outputs

            s_outputs = s_outputs.transpose(2, 1)
            sent_topics = self.gen_topic(s_outputs)
            # for i in range(5):
            #     print (torch.dist(sent_topics[0, i+1, :], sent_topics[0, 0, :]))
            pred_stops = self.stop_classifier(sent_topics)

            topics = sent_topics[:, sent_id:sent_id + 1, :]

            global_inputs = torch.cat([topics, start_words,
                                       start_words.new_zeros((real_bts, self.w_max - 1, self.emb_size))], 1)
            global_inputs = global_inputs.transpose(2, 1)

            res_words_idx = torch.ones((real_bts, self.w_max), dtype=torch.int)

            state = [[res_words_idx.clone(), global_inputs.clone(), [0] * real_bts] for _ in range(beam_size)]

            for j in range(1, self.w_max + 1):
                tmp = [[{'log_prob': 0, 'cur_idx': 0, 'prev_idx': 0} for _ in range(beam_size * beam_size)] for _ in
                       range(real_bts)]
                for beam_i in range(beam_size):

                    w_inputs = state[beam_i][1]
                    for i, conv in enumerate(self.wCNN):
                        res = w_inputs
                        w_outputs = conv(w_inputs)
                        w_outputs = w_outputs[:, :, :-self.w_padding]
                        w_outputs = F.glu(w_outputs, dim=1)
                        w_outputs = (w_outputs + res) * math.sqrt(.5)
                        w_inputs = w_outputs
                    now_words = self.classifier(w_outputs[:, :, j])
                    now_words = self.softmax(now_words).cpu()
                    for beam_j in range(beam_size - 1, -1, -1):
                        j_th_max = torch.kthvalue(-now_words, beam_size - beam_j)
                        jth_prob = -j_th_max[0]
                        jth_idx = j_th_max[1]
                        for k in range(real_bts):
                            tmp[k][beam_i + beam_j * beam_size]['log_prob'] = state[beam_i][2][k] + math.log(
                                float(jth_prob[k]) + 1e-9)
                            tmp[k][beam_i + beam_j * beam_size]['cur_idx'] = int(jth_idx[k])
                            tmp[k][beam_i + beam_j * beam_size]['prev_idx'] = beam_i

                new_state = [[res_words_idx.clone(), global_inputs.clone(), [0] * real_bts] for _ in range(beam_size)]
                for k in range(real_bts):
                    tmp[k].sort(key=lambda p: p['log_prob'], reverse=True)
                    if j == 1:
                        for o in range(1, beam_size):
                            tmp[k][o], tmp[k][o + beam_size * o] = tmp[k][o + beam_size * o], tmp[k][o]
                    for beam_i in range(beam_size):
                        prev_beam_idx = tmp[k][beam_i]['prev_idx']
                        new_state[beam_i][0][k] = state[prev_beam_idx][0][k].clone()
                        new_state[beam_i][0][k, j - 1] = tmp[k][beam_i]['cur_idx']
                        new_state[beam_i][2][k] = tmp[k][beam_i]['log_prob']
                        if j != self.w_max:
                            new_state[beam_i][1][k] = state[prev_beam_idx][1][k].clone()
                            new_state[beam_i][1][k, :, j + 1] = self.embedding(
                                new_state[beam_i][0][k, j - 1].long().to(self.device))

                state = new_state

            res_words_idx = [list(i) for i in list(state[0][0].numpy())]
            prev_sents = torch.LongTensor(res_words_idx).to(self.device)
            res_words = [[self.idx2word[int(j)] for j in i] for i in res_words_idx]
            prev_mask_sum = []

            for k in range(real_bts):
                end_pos = res_words[k].index('<eos>') if '<eos>' in res_words[k] else self.w_max + 1
                prev_mask_sum.append(end_pos + 1)
                prev_sents[k][end_pos + 1:] = 2
                result[k] += ' '.join(res_words[k][:end_pos] + ['. '])

            prev_sents_embed = self.embedding(prev_sents)
            prev_mask_sum = torch.Tensor(prev_mask_sum).to(self.device).unsqueeze(1)
            prev_sents_embed = torch.max(prev_sents_embed, 1)[0]

            stop_flag = self.softmax(self.stop_classifier(sRNN_output.squeeze(0)))

            if float(stop_flag[0, 1]) >= 0.5:
                break

        return [r.strip() for r in result]

    def sample(self, img_feats):

        img_feats = img_feats.view(-1, self.num_boxes, img_feats.size(-1))

        real_bts = img_feats.size(0)

        proj_feats = self.project(img_feats)
        pool_feats = torch.max(proj_feats, 1)[0].view(real_bts, 1, self.proj_size)

        result = ['' for _ in range(real_bts)]

        s_inputs = pool_feats.expand(real_bts, self.s_max, self.emb_size)
        s_inputs = s_inputs.transpose(2, 1)
        for i, conv in enumerate(self.sCNN):
            if i != 0:
                att_weight = torch.bmm(proj_feats, self.att_proj(s_inputs.transpose(2, 1)).transpose(2, 1))
                att_weight = nn.Softmax(1)(att_weight)
                s_inputs = torch.bmm(proj_feats.permute(0, 2, 1), att_weight) + s_inputs
            res = s_inputs
            s_outputs = conv(s_inputs)
            s_outputs = s_outputs[:, :, :-self.s_padding]
            s_outputs = F.glu(s_outputs, dim=1)
            s_outputs = (s_outputs + res) * math.sqrt(0.5)
            s_inputs = s_outputs

        s_outputs = s_outputs.transpose(2, 1)
        sent_topics = self.gen_topic(s_outputs)
        pred_stops = self.stop_classifier(sent_topics)

        for sent_id in range(self.s_max):

            topics = sent_topics[:, sent_id:sent_id + 1, :]
            start_words = torch.zeros((real_bts, 1), dtype=torch.long, device=self.device)
            start_words = self.embedding(start_words)
            global_inputs = torch.cat([topics, start_words,
                                       start_words.new_zeros(real_bts, self.w_max - 1, self.emb_size)], 1)
            global_inputs = global_inputs.transpose(2, 1)

            res_words_idx = torch.zeros((real_bts, self.w_max), dtype=torch.int)
            for j in range(1, self.w_max + 1):
                w_inputs = global_inputs
                for i, conv in enumerate(self.wCNN):
                    res = w_inputs
                    w_outputs = conv(w_inputs)
                    w_outputs = w_outputs[:, :, :-self.w_padding]
                    w_outputs = F.glu(w_outputs, dim=1)
                    w_outputs = (w_outputs + res) * math.sqrt(.5)
                    w_inputs = w_outputs
                now_words = self.classifier(w_outputs[:, :, j])
                now_words = torch.max(now_words, 1)[1]
                res_words_idx[:, j - 1] = now_words
                now_words = self.embedding(now_words)
                if j != self.w_max:
                    global_inputs[:, :, j + 1] = now_words
            res_words_idx = [list(i) for i in list(res_words_idx.numpy())]
            res_words = [[self.idx2word[int(j)] for j in i] for i in res_words_idx]
            for k in range(real_bts):
                end_pos = res_words[k].index('<eos>') if '<eos>' in res_words[k] else self.w_max + 1
                result[k] += ' '.join(res_words[k][:end_pos] + ['. '])

                stop_flag = self.softmax(self.stop_classifier(sRNN_output.squeeze(0)))

                if float(stop_flag[0, 1]) >= 0.5:
                    break

        return [r.strip() for r in result]


net = DualCNN()
param = net.parameters()
for i in param:
    print (i.size())
