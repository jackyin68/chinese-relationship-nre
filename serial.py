import pickle as pkl
import numpy as np
import os
import tensorflow as tf
import logging

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('raw_data_path', 'data/raw', 'raw data dir to load')
tf.app.flags.DEFINE_string('processed_data_path', 'data/processed', 'processed data dir to load')
tf.app.flags.DEFINE_string('model_path', 'data/model', 'save model dir')

tf.app.flags.DEFINE_string('cuda', '0', 'gpu id')
tf.app.flags.DEFINE_boolean('pre_embed', True, 'load pre-trained word2vec')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('epochs', 200, 'max train epochs')
tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of hidden embedding')
tf.app.flags.DEFINE_integer('word_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('pos_dim', 5, 'dimension of position embedding')
tf.app.flags.DEFINE_integer('pos_limit', 15, 'max distance of position embedding')
tf.app.flags.DEFINE_integer('sen_len', 60, 'sentence length')
tf.app.flags.DEFINE_integer('window', 3, 'window size')
tf.app.flags.DEFINE_string('level', 'bag', 'bag level or sentence level, option:bag/sent')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout rate')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('word_frequency', 5, 'minimum word frequency when constructing vocabulary list')

logger = logging.getLogger('Serial')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def dump_pkl(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)
    f.close()


def pos_index(x, pos_limit):
    if x < -pos_limit:
        return 0
    if -pos_limit <= x <= pos_limit:
        return x + pos_limit + 1
    if x > pos_limit:
        return 2 * pos_limit + 2


def load_sent(filename, word_map, flags):
    logger.info('load_sent...', filename)
    sentence_dict = {}
    with open(filename, 'r') as fr:
        for line in fr:
            id_, en1, en2, sentence = line.strip().split('\t')
            print("load_sent: ", sentence)
            word_list = sentence.split()
            en1_pos = 0
            en2_pos = 0
            for i in range(len(word_list)):
                if word_list[i] == en1:
                    en1_pos = i
                if word_list[i] == en2:
                    en2_pos = i

            words = []
            pos1 = []
            pos2 = []
            length = min(flags.sen_len, len(word_list))
            for i in range(length):
                words.append(word_map.get(word_list[i], word_map['UNK']))
                pos1.append(pos_index(i - en1_pos, flags.pos_limit))
                pos2.append(pos_index(i - en2_pos, flags.pos_limit))

            if length < flags.sen_len:
                for i in range(length, flags.sen_len):
                    words.append(word_map['PAD'])
                    pos1.append(pos_index(i - en1_pos, flags.pos_limit))
                    pos2.append(pos_index(i - en2_pos, flags.pos_limit))
            sentence_dict[id_] = np.reshape(np.asarray([words, pos1, pos2], dtype=np.int32), (1, 3, flags.sen_len))
        return sentence_dict


def trans2ids(sentence_dict, level, relation_file, out_path, set_type, num_classes=35):
    logger.info('trans2ids...{}-{}'.format(level, set_type))
    if level == 'bag':
        all_bags = []
        all_sents = []
        all_labels = []
        with open(relation_file, 'r') as fr:
            idebug = 0
            for line in fr:
                print("trans2ids: ", level, set_type, line)
                rel = [0] * num_classes
                print("trans2ids: ", idebug)
                idebug = idebug + 1
                try:
                    bag_id, _, _, sents, types = line.strip().split('\t')
                    type_list = types.split()
                    for tp in type_list:
                        if len(type_list) > 1 and tp == '0':
                            # if a bag has multiple relations, we only consider non-NA relations
                            continue
                        rel[int(tp)] = 1
                except:
                    bag_id, _, _, sents = line.strip().split('\t')

                sent_list = []
                if len(sents.split()) > 1:
                    print(len(sents.split()))
                for sent in sents.split():
                    sent_list.append(sentence_dict[sent])

                all_bags.append(bag_id)
                all_sents.append(np.concatenate(sent_list, axis=0))
                all_labels.append(np.asarray(rel, dtype=np.float32))
        out_path = os.path.join(out_path, set_type, level)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        dump_pkl(os.path.join(out_path, 'all_bags.pkl'), all_bags)
        dump_pkl(os.path.join(out_path, 'all_sents.pkl'), all_sents)
        dump_pkl(os.path.join(out_path, 'all_labels.pkl'), all_labels)
    else:
        all_sent_ids = []
        all_sents = []
        all_labels = []
        with open(relation_file, 'r') as fr:
            for line in fr:
                print("trans2ids: ", level, set_type, line)
                rel = [0] * num_classes
                try:
                    sent_id, types = line.strip().split('\t')
                    type_list = types.split()
                    for tp in type_list:
                        if len(type_list) > 1 and tp == '0':
                            # if a sentence has multiple relations, we only consider non-NA relations
                            continue
                        rel[int(tp)] = 1
                except:
                    sent_id = line.strip()

                all_sent_ids.append(sent_id)
                all_sents.append(sentence_dict[sent_id])
                all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, num_classes)))

        all_sents = np.concatenate(all_sents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        out_path = os.path.join(out_path, set_type, level)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        dump_pkl(os.path.join(out_path, 'all_sent_ids.pkl'), all_sent_ids)
        dump_pkl(os.path.join(out_path, 'all_sents.pkl'), all_sents)
        dump_pkl(os.path.join(out_path, 'all_labels.pkl'), all_labels)


def create_serial(flags):
    logger.info('create_serial')
    levels = ['bag', 'sent']
    set_types = ['train', 'dev', 'test']
    with open(os.path.join(flags.processed_data_path, 'word_map.pkl'), 'rb') as fm:
        word_map = pkl.load(fm)

    for set_type in set_types:
        print('Transforming {} sets'.format(set_type))
        sent = load_sent(os.path.join(flags.raw_data_path, 'sent_' + set_type + '.txt'), word_map, flags)
        for level in levels:
            print('In level {}'.format(level))
            trans2ids(sent, level, os.path.join(flags.raw_data_path, level + '_relation_' + set_type + '.txt'),
                      flags.processed_data_path, set_type)


def main(_):
    create_serial(FLAGS)


if __name__ == '__main__':
    tf.app.run()
