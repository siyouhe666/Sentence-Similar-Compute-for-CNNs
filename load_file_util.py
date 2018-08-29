from sentences_generator import sentences_generator
import gensim
import numpy as np
import math
###NOTE###
#Author:siyouhe666@gmail.com
# please update this file so that you can match your train or test file format.#
def get_data_list(path):
    data_list = sentences_generator(path)
    return data_list


def load_w2vmodel(path):
    w2vmodel = gensim.models.Word2Vec.load(path)
    return w2vmodel


def sen2vec(sen, w2vmodel, fixed_height):
    new_sen = []
    for word in sen:
        if word in w2vmodel.wv:
            new_sen.append(word)
    len_sen = len(new_sen)
    if len_sen == 0:
        sen_matrix = np.zeros((fixed_height, 100))
    elif len_sen > fixed_height:
        new_sen = new_sen[:fixed_height]
        sen_matrix = w2vmodel[new_sen]
    else:
        append_num = fixed_height - len_sen
        sen_matrix = w2vmodel[new_sen]
        z_matrix = np.zeros((append_num, 100))
        sen_matrix = np.append(sen_matrix, z_matrix, axis=0)
    return sen_matrix


def load_dataset(input_path, w2vmodel_path, fixed_height, align_sentence=False):
    print("载入数据...")
    X = []
    Y = []
    sen1_list = []
    sen2_list = []
    data_list = get_data_list(input_path)
    w2vmodel = load_w2vmodel(w2vmodel_path)
    for line in data_list:
        no, sen1, sen2, label = line
        if align_sentence:
            sen1, sen2 = align_sen(sen1, sen2)
        sen1_matrix = sen2vec(sen1, w2vmodel, fixed_height)
        sen2_matrix = sen2vec(sen2, w2vmodel, fixed_height)
        sen1_matrix = build_matrix(sen1_matrix, 1)
        sen2_matrix = build_matrix(sen2_matrix, 1)
        sen1_list.append(sen1_matrix)
        sen2_list.append(sen2_matrix)
        Y.append(int(label))
    X.append(sen1_list)
    X.append(sen2_list)
    X = np.array(X)
    Y = np.array(Y)
    Y = convert_to_one_hot(Y, 2).T
    print("载入完成")
    return X, Y


def build_matrix(matrix, channel_num):
    height, width = matrix.shape
    new_matrix = np.zeros((height, width, channel_num))
    for i in range(height):
        for j in range(width):
            new_matrix[i][j][0] = matrix[i][j]
    return new_matrix


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def align_sen(sen1, sen2):
    if len(sen1) < len(sen2):
        temp = sen1
        sen1 = sen2
        sen2 = temp
    sen1 = sen1.split(" ")
    sen2 = sen2.split(" ")
    same = []
    for word in sen1:
        if word in sen2:
            same.append(word)
    new_sen1, new_sen2 = [], []
    for word in same:
        if word in sen1:
            sen1.remove(word)
            new_sen1.append(word)
        if word in sen2:
            sen2.remove(word)
            new_sen2.append(word)
    new_sen1.extend(sen1)
    new_sen2.extend(sen2)
    return new_sen1, new_sen2


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches