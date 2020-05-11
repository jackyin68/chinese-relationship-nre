import os
import pickle as pkl

processed_data_path = 'data/processed'


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        obj = pkl.load(f)
    f.close()
    return obj


def load_set(set_type):
    print('Loading {} sets'.format(set_type))
    path = os.path.join(processed_data_path, set_type, 'sent')
    sents = load_pkl(os.path.join(path, 'all_sents_embed.pkl'))
    labels = load_pkl(os.path.join(path, 'all_labels_embed.pkl'))
    return sents, labels


def load_data():
    (x_train, y_train) = load_set('train')
    (x_test, y_test) = load_set('test')

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    load_data()
