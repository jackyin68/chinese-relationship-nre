import logging
import multiprocessing
import re
import string
import jieba
from tqdm import tqdm

from zhon.hanzi import punctuation

raw_file = 'data/raw/text.txt'
clean_file = 'data/processed/clean_text.txt'
seg_file = 'data/processed/seg_text.txt'

logger = logging.getLogger('Preprocess')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def clean_func(line):
    line_punct_replaced = re.sub(r"[%s]+" % punctuation, " ", line)
    line_string_punct_replaced = re.sub(r"[%s]+" % string.punctuation, " ", line_punct_replaced)
    line_removed_character = re.sub(r"[a-zA-Z]+", " ", line_string_punct_replaced)
    return line_removed_character


def data_clean():
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    logger.info('Loading raw text')
    with open(raw_file, 'r', encoding='utf8') as fin:
        raw_lines = fin.readlines()
    fin.close()

    logger.info('Cleaning raw text')
    results = []
    lines = tqdm(raw_lines, desc=u'已清理0行文本')
    for i, line in enumerate(lines):
        results.append(pool.apply(clean_func, (line,)))
        if i % 10000 == 0:
            lines.set_description(u'已清理%s行文本' % i)

    pool.close()
    pool.join()

    logger.info('Writing clean text')
    with open(clean_file, 'w', encoding='utf8') as fout:
        line_min_len = 5
        for line in results:
            if len(line.strip()) < line_min_len:
                continue
            fout.writelines(line)
    fout.close()
    logger.info('Text clean ended')


def seq_func(line):
    # print("seq:", line)
    return " ".join(jieba.cut(line, cut_all=False))


def data_seg():
    logger.info('Segmenting cleaned text')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    logger.info('Loading clean text')
    with open(clean_file, 'r', encoding='utf8') as fin:
        clean_lines = fin.readlines()
    fin.close()

    logger.info('Segmenting clean text')
    results = []
    lines = tqdm(clean_lines, desc=u'已完成0行文本分词')
    for i, line in enumerate(lines):
        results.append(pool.apply(seq_func, (line,)))
        if i % 10000 == 0:
            lines.set_description(u'已完成%s行文本分词' % i)

    pool.close()
    pool.join()

    logger.info('Writing segment text')
    with open(seg_file, 'w', encoding='utf8') as fout:
        for line in results:
            fout.writelines(line)
    fout.close()
    logger.info('Text segement ended')


if __name__ == "__main__":
    data_clean()
    data_seg()
