import os
import pickle as pkl
from collections import Counter
import numpy as np


def dump_pkl(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)
    f.close()


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        obj = pkl.load(f)
    f.close()
    return obj


def load_wordVec(self):
    word_map = load_pkl(os.path.join(self.processed_data_path, 'word_map.pkl'))
    word_embed = load_pkl(os.path.join(self.processed_data_path, 'word_embed.pkl'))
    return word_map, word_embed


def load_wordMap(self):
    word_map = {}
    word_map['PAD'] = len(word_map)
    word_map['UNK'] = len(word_map)
    all_content = []
    for line in open(os.path.join(self.raw_data_path, 'sent_train.txt')):
        all_content += line.strip().split('\t')[3].split()
    for item in Counter(all_content).most_common():
        if item[1] > self.word_frequency:
            word_map[item[0]] = len(word_map)
        else:
            break
    return word_map


def load_relation(self):
    relation2id = {}
    for line in open(os.path.join(self.raw_data_path, 'relation2id.txt')):
        relation, id_ = line.strip().split()
        relation2id[relation] = int(id_)
    return relation2id


def load_set(self, set_type):
    all_pkls = []
    print('Loading {} sets'.format(set_type))
    path = os.path.join(self.processed_data_path, set_type, 'bag' if self.bag else 'sent')
    for file in os.listdir(path):
        all_pkls.append(load_pkl(os.path.join(path, file)))
    print('Num of samples - {}'.format(len(all_pkls[-1])))
    return all_pkls


def data_batcher(self, all_files, padding=False, shuffle=True):
    if self.bag:
        all_sents = all_files[0]
        all_bags = all_files[1]
        all_labels = all_files[2]

        self.data_size = len(all_bags)
        self.datas = all_bags
        data_order = list(range(self.data_size))
        if shuffle:
            np.random.shuffle(data_order)
        if padding:
            if self.data_size % self.batch_size != 0:
                data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

        for i in range(len(data_order) // self.batch_size):
            total_sens = 0
            out_sents = []
            out_sent_nums = []
            out_labels = []
            for k in data_order[i * self.batch_size:(i + 1) * self.batch_size]:
                out_sents.append(all_sents[k])
                out_sent_nums.append(total_sens)
                total_sens += all_sents[k].shape[0]
                out_labels.append(all_labels[k])

            out_sents = np.concatenate(out_sents, axis=0)
            out_sent_nums.append(total_sens)
            out_sent_nums = np.asarray(out_sent_nums, dtype=np.int32)
            out_labels = np.stack(out_labels)

            yield out_sents, out_labels, out_sent_nums
    else:
        all_sent_ids = all_files[0]
        all_sents = all_files[1]
        all_labels = all_files[2]

        self.data_size = len(all_sent_ids)
        self.datas = all_sent_ids
        data_order = list(range(self.data_size))
        if shuffle:
            np.random.shuffle(data_order)
        if padding:
            if self.data_size % self.batch_size != 0:
                data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

        for i in range(len(data_order) // self.batch_size):
            idx = data_order[i * self.batch_size:(i + 1) * self.batch_size]
            yield all_sents[idx], all_labels[idx], None
