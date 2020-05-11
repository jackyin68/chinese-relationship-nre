import logging
import multiprocessing
import os
import pickle as pkl
import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from gensim.models.word2vec import PathLineSentences

logger = logging.getLogger('Word2Vec')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

seg_file = 'data/processed/seg_text.txt'
word_vec_file = 'data/processed/word2vec.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('processed_data_path', 'data/processed', 'processed data dir to load')
tf.app.flags.DEFINE_integer('word_dim', 300, 'dimension of word embedding')


def word_vec():
    logger.info('Word to vec')
    model = word2vec.Word2Vec(PathLineSentences(seg_file), sg=1, size=300, window=5, min_count=10, sample=1e-4,
                              workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format(word_vec_file, binary=False)


def dump_pkl(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)
    f.close()


def create_word_vec(flags):
    logger.info('Word map and embedding')
    word_map = {}
    word_map['PAD'] = len(word_map)
    word_map['UNK'] = len(word_map)
    word_embed = []
    for line in open(word_vec_file, 'r'):
        content = line.strip().split()
        if len(content) != flags.word_dim + 1:
            continue
        word_map[content[0]] = len(word_map)
        word_embed.append(np.asarray(content[1:], dtype=np.float32))

    word_embed = np.stack(word_embed)
    embed_mean, embed_std = word_embed.mean(), word_embed.std()

    pad_embed = np.random.normal(embed_mean, embed_std, (2, flags.word_dim))
    word_embed = np.concatenate((pad_embed, word_embed), axis=0)
    word_embed = word_embed.astype(np.float32)
    print('Word in dict - {}'.format(len(word_map)))

    dump_pkl(os.path.join(flags.processed_data_path, 'word_map.pkl'), word_map)
    dump_pkl(os.path.join(flags.processed_data_path, 'word_embed.pkl'), word_embed)


def main(_):
    word_vec()
    create_word_vec(FLAGS)


if __name__ == "__main__":
    tf.app.run()
