import os
import pickle as pkl
import numpy as np
from copy import deepcopy
import gc
from tqdm import tqdm
import objgraph
import sys

processed_data_path = 'data/processed'
word_dim = 300
sen_len = 60
pos_limit = 15
pos_num = 2 * pos_limit + 3
train_num_limit = 10000
test_num_limit = 2000


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        obj = pkl.load(f)
    f.close()
    return obj


def dump_pkl(file_path, obj):  # 写入对象保存到pkl文件
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f, protocol=4)
    f.close()


def load_wordvec():
    word_map = load_pkl(os.path.join(processed_data_path, 'word_map.pkl'))  # 读取单词索引
    word_embed = load_pkl(os.path.join(processed_data_path, 'word_embed.pkl'))  # 读取单词词向量
    return word_map, word_embed


def pos_embedding(pos):
    bin_str = bin(pos).replace('0b', '')
    bin_str_list = list(bin_str.rjust(6, '0'))
    bin_num_list = list(map(int, bin_str_list))
    return bin_num_list


word_map, word_embed = load_wordvec()
sent_embedding_global = np.empty(shape=[60, 312], dtype=float)


def get_sents(sent_file_path):
    sents = load_pkl(sent_file_path)
    for sent in tqdm(sents):
        yield sent


# @profile
def sents_to_vec(sent_file_path, set_type):
    i_debug = 0
    num_limit = train_num_limit if set_type == 'train' else test_num_limit
    sents_embedding = []
    sent_embedding = np.empty(shape=[60, 312], dtype=float)
    sents = get_sents(sent_file_path)
    for sent in sents:
        if i_debug >= num_limit:
            break
        i_debug = i_debug + 1

        for sen_word_index in range(sen_len):
            word_embedding = word_embed[sent[0][sen_word_index]]
            word1_pos_embedding = pos_embedding(sent[1][sen_word_index])
            word2_pos_embedding = pos_embedding(sent[2][sen_word_index])

            word_embeddings = []
            word_embeddings.extend(word_embedding)
            word_embeddings.extend(word1_pos_embedding)
            word_embeddings.extend(word2_pos_embedding)
            sent_embedding[sen_word_index, :] = np.array(word_embeddings, ndmin=2)
        sents_embedding.append(deepcopy(sent_embedding))
    return sents_embedding


def sents_to_vec_ls(sent_file_path, set_type):
    i_debug = 0
    num_limit = train_num_limit if set_type == 'train' else test_num_limit
    sents_embedding = []
    sent_embedding = np.empty(shape=[60, 312], dtype=float)
    sents = get_sents(sent_file_path)
    for sent in sents:
        if i_debug > num_limit:
            break
        i_debug = i_debug + 1

        for sen_word_index in range(sen_len):
            word_embedding = word_embed[sent[0][sen_word_index]]
            word1_pos_embedding = pos_embedding(sent[1][sen_word_index])
            word2_pos_embedding = pos_embedding(sent[2][sen_word_index])

            word_embeddings = []
            word_embeddings.extend(word_embedding)
            word_embeddings.extend(word1_pos_embedding)
            word_embeddings.extend(word2_pos_embedding)

            sent_embedding[sen_word_index] = np.array(word_embeddings, ndmin=2)
        sents_embedding.append(deepcopy(sent_embedding))
    return np.array(sents_embedding)

# @profile
def sent_embed_creat(set_type):
    print('Loading {} sets'.format(set_type))
    dir_path = os.path.join(processed_data_path, set_type, 'sent')
    sent_file_path = os.path.join(dir_path, 'all_sents.pkl')
    sents_vec = sents_to_vec(sent_file_path, set_type)
    sents_vec_nr = np.array(deepcopy(sents_vec))

    sent_dir_file_path = os.path.join(dir_path, 'all_sents_embed.pkl')
    dump_pkl(sent_dir_file_path, sents_vec_nr)


def labels_to_vec(labels, set_type):
    num_limit = train_num_limit if set_type == 'train' else test_num_limit
    return labels[:num_limit, :]


def label_embed_creat(set_type):
    print('Loading {} sets'.format(set_type))
    path = os.path.join(processed_data_path, set_type, 'sent')
    labels = load_pkl(os.path.join(path, 'all_labels.pkl'))
    labels_vec = labels_to_vec(labels, set_type)
    dump_pkl(os.path.join(path, 'all_labels_embed.pkl'), labels_vec)


if __name__ == "__main__":
    sent_embed_creat('train')
    sent_embed_creat('test')
    label_embed_creat('train')
    label_embed_creat('test')
