# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import json

data_dir = 'data'

origin_path = os.path.join(data_dir, 'paragraphs_v1.json')
img2sents_path = os.path.join(data_dir, 'img2sents.pkl')
img2dense_path = os.path.join(data_dir, 'img2dense.json')
img2onehot_path = os.path.join(data_dir, 'img2onehot.pkl')
# img2densevec_path = os.path.join(data_dir, 'img2dense_vec.pkl')
vocab_path = os.path.join(data_dir, 'vocab.pkl')

def get_img2sents(update_flag=False):
    if os.path.exists(img2sents_path) and not update_flag:
        return
    
    origin_data = json.load(open(origin_path, 'r'))
    img2paragraph = {}
    
    for each_data in origin_data:
        image_id = each_data['image_id']
        paragraph = each_data['paragraph']
        paragraph = paragraph.replace('t.v.', 'tv').replace('U.S.', 'US').replace('T.C.', 'TC').replace(
            'C.E.T.', 'CET')
        paragraph = paragraph.replace(' st.', ' st').replace(' ST.', ' ST').replace(' Mt. ', ' Mt ').replace(
            ' St.', ' St').replace(' Dept. ', ' Dept ')
        paragraph = paragraph.replace(' S. ', ' st ').replace('welcomebackveterans.org.',
                                                                        'welcomebackveterans')
        paragraph = paragraph.replace('$1.00', '$1').replace('3.20', '320').replace('us.open.org',
                                                                                              'usopenorg')
        paragraph = paragraph.replace('evil. ECC. IV.23', 'evilECCIV23').replace('neweracap.com',
                                                                                           'neweracapcom')
        paragraph = paragraph.replace(' Baby toys. And boxes.', ' ').replace('$1.25', '$1').replace(
            '.UMBRELLA.', 'UMBRELLA.')
        paragraph = paragraph.replace('www.kiwirail.co.nz', 'wwwkiwirailconz').replace('28.41',
                                                                                                 '2841').replace(
            'http://www.tmz.com/', '')
        paragraph = paragraph.replace(' Handle. ', ' Handle ').replace(' oz. ', ' oz ')
        paragraph = paragraph.replace('CapeTreasures.com', 'CapeTreasurescom').replace('NW Meadow... DR.',
                                                                                                 'DR')
        paragraph = paragraph.replace('$.20', '$1').replace('www.theimpusilvebuy.com',
                                                                      'wwwtheimpusilvebuycom')
        paragraph = paragraph.replace('transavia.com', 'transaviacom').replace('XL.com', 'XLcom')

        paragraph.replace(' .', '.')
        paragraph.replace('. ', '.')
        sentences = paragraph.split('.')
        sentences = map(lambda sent: sent.strip(), sentences)
        sentences = filter(lambda sent: len(sent) >= 2, sentences)
        # if sentences!=[]:
        #     print sentences
        # if '.cn' in paragraph:
        #     print paragraph

        img2paragraph[image_id] = sentences

    pickle.dump(img2paragraph, open(img2sents_path, 'wb'))


def get_vocab(word_count_threshold=5, update_flag=False):
    if os.path.exists(vocab_path) and not update_flag:
        return

    img2para = pickle.load(open(img2sents_path, 'rb'))
    all_sents = []
    for key, para in img2para.items():
        for each_sent in para:
            each_sent = each_sent.replace(',', ' , ')
            all_sents.append(each_sent)

    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold,))

    word_counts = {}
    nsents = 0

    for sent in all_sents:
        nsents += 1
        tmp_sent = sent.lower().split(' ')

        for w in tmp_sent:
            if w != '' and w != ' ':
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    idx2word = {}
    idx2word[0] = '<bos>'
    idx2word[1] = '<eos>'
    idx2word[2] = '<pad>'
    idx2word[3] = '<unk>'

    word2idx = {}
    word2idx['<bos>'] = 0
    word2idx['<eos>'] = 1
    word2idx['<pad>'] = 2
    word2idx['<unk>'] = 3

    for idx, w in enumerate(vocab):
        word2idx[w] = idx + 4
        idx2word[idx + 4] = w

    word_counts['<eos>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<pad>'] = nsents
    word_counts['<unk>'] = nsents

    # bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    # bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    # bias_init_vector = np.log(bias_init_vector)
    # bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    pickle.dump([word2idx, idx2word], open(vocab_path, 'wb'))


def get_one_hot(s_max, n_max, update_flag):
    if os.path.exists(img2onehot_path) or not update_flag:
        return

    word2idx = pickle.load(open(vocab_path, 'rb'))[0]
    img2para = pickle.load(open(img2sents_path, 'rb'))
    img2para_vec = {}

    for img, para in img2para.items():

        num_sents = min(len(para), s_max)
        sent_stop = np.zeros(s_max, dtype=int)
        sent_stop[num_sents - 1:] = 1
        paras_idx = np.ones([s_max, n_max + 1], dtype=int) * 2

        for sent_id, sent in enumerate(para):
            if sent_id == num_sents:
                break
            sent = sent.replace(',', ' , ')

            sent = '<bos> ' + sent + ' <eos>'
            word_count = 0
            tmp_sent = sent.lower().split(' ')
            tmp_sent = filter(lambda x: x != '' and x != ' ', tmp_sent)

            for word_id, word in enumerate(tmp_sent):
                if word_id == n_max + 1:
                    break

                word_count += 1
                if word in word2idx:
                    paras_idx[sent_id, word_id] = word2idx[word]
                else:
                    paras_idx[sent_id, word_id] = word2idx['<unk>']

        img2para_vec[str(img)] = [paras_idx, sent_stop]

    pickle.dump(img2para_vec, open(img2onehot_path, 'wb'))

# def getDenseVec():
#     img2dense = json.load(open(img2dense_path, 'r'))
#     img2dense_vec = {}
#     for img, captions in img2dense.items():
#         dense_vec = np.ones((50, 6), dtype=int) * 2
#         for i, caption in enumerate(captions):
#             if i >= 50:
#                 break
#             words = caption.split(' ')
#             for j, word in enumerate(words):
#                 if j >= 6:
#                     break
#                 if word in word2idx:
#                     dense_vec[i, j] = word2idx[word]
#                 else:
#                     dense_vec[i, j] = word2idx['<unk>']
#         img2dense_vec[img] = dense_vec
#     with open(img2densevec_path, 'wb') as f:
#         pickle.dump(img2dense_vec, f)

def run(update_flag):
    get_img2sents(update_flag)
    get_vocab(update_flag=update_flag)
    get_one_hot(6, 30, update_flag=update_flag)
    # getDenseVec()

    print('Data preprocess done')

if __name__ == '__main__':
    run(True)