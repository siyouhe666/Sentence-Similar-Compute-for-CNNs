import tensorflow as tf
from load_file_util import *
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time

#Author:siyouhe666@gmail.com
#this file is the most important file contains algorithm and file processing program
#you can use this file to train your model through replace the word2vec model and train data.
#word2vec model file and train data was put in other floders, please check it yourself.


#################similarity compute between two vector#################
"""
def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d


def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d


def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        #cosine=x*y/(|x||y|)
        #先求x，y的模 #|x|=sqrt(x1^2+x2^2+...+xn^2)
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)) #reduce_sum函数在指定维数上进行求和操作
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        #求x和y的内积
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        #内积除以模的乘积
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d

def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y), compute_l1_distance(x, y)]
    #stack函数是将list转化为Tensor
    return tf.stack(result, axis=1)

def comU2(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)
"""


def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d


def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1)
        return d


def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        #cosine=x*y/(|x||y|)
        #先求x，y的模 #|x|=sqrt(x1^2+x2^2+...+xn^2)
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)) #reduce_sum函数在指定维数上进行求和操作
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        #求x和y的内积
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        #内积除以模的乘积
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d

def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y), compute_l1_distance(x, y)]
    #stack函数是将list转化为Tensor
    return tf.stack(result, axis=1)

def comU2(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)
#################计算相似度#################


def initialize_parameters(channel_num, embedding_dim):
    filter_size = [1, 2, 30]
    filter_num = [20, 20]
    poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]
    tf.set_random_seed(1)
    W1 = [tf.get_variable("W11gram", [filter_size[0], embedding_dim, channel_num, filter_num[0]], initializer=tf.contrib.layers.xavier_initializer(seed=0)),
          tf.get_variable("W12gram", [filter_size[1], embedding_dim, channel_num, filter_num[0]], initializer=tf.contrib.layers.xavier_initializer(seed=0)),
          tf.get_variable("W1ngram", [filter_size[2], embedding_dim, channel_num, filter_num[0]], initializer=tf.contrib.layers.xavier_initializer(seed=0))]

    B1 = [tf.Variable(tf.constant(0.1, shape=[filter_num[0]])),
          tf.Variable(tf.constant(0.1, shape=[filter_num[0]])),
          tf.Variable(tf.constant(0.1, shape=[filter_num[0]]))]

    W2 = [tf.get_variable("W21gram", [filter_size[0], embedding_dim, channel_num, filter_num[1]], initializer=tf.contrib.layers.xavier_initializer(seed=0)),
          tf.get_variable("W22gram", [filter_size[1], embedding_dim, channel_num, filter_num[1]], initializer=tf.contrib.layers.xavier_initializer(seed=0))]

    B2 = [tf.Variable(tf.constant(0.1, shape=[filter_num[1], embedding_dim])),
               tf.Variable(tf.constant(0.1, shape=[filter_num[1], embedding_dim]))]

    parameters = {"W1": W1,
                  "B1": B1,
                  "W2": W2,
                  "B2": B2,
                  "filter_size": filter_size,
                  "filter_num": filter_num,
                  "poolings": poolings
                  }
    return parameters


def bulit_block_A(x, parameters):
    # bulid block A and cal the similarity according to algorithm 1
    poolings = parameters['poolings']
    filter_sizes = parameters['filter_size']
    W1 = parameters['W1']
    B1 = parameters['B1']
    out = []
    with tf.name_scope("bulid_block_A"):
        # 遍历每个pooling方式，max、min、mean
        for pooling in poolings:
            pools = []
            # 每个pooling都对应几种不同的窗口大小。【1，2，100】
            for i, ws in enumerate(filter_sizes):
                # print x.get_shape(), W1[i].get_shape()
                with tf.name_scope("conv-pool-%s" % ws):
                    # x->[batch_size, sentence_length, embed_size, 1], W1[i]->[ws, embed_size, 1, num_filters]
                    conv = tf.nn.conv2d(x, W1[i], strides=[1, 1, 1, 1], padding="VALID")
                    # print conv.get_shape()
                    # conv = tf.nn.relu(conv + B1[i])  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
                    conv = tf.nn.tanh(conv + B1[i])
                    pool = pooling(conv, axis=1)  ## [batch_size, 1, num_filters_A]
                pools.append(pool)
            out.append(pools)
    return out

def per_dim_conv_layer(x, w, b, pooling):
    '''
    :param input: [batch_size, sentence_length, embed_size, 1]
    :param w: [ws, embedding_size, 1, num_filters]
    :param b: [num_filters, embedding_size]
    :param pooling:
    :return:
    '''
    # 为了实现per_dim的卷积。所以我们要将输入和权重偏置参数在embed_size维度上进行unstack
    # 这样我们就获得了每个维度上的输入、权重、偏置。可以结合模型介绍篇里面的图片进行理解
    input_unstack = tf.unstack(x, axis=2) #对wordembedding维度进行拆解
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(b, axis=1)
    convs = []
    # 对每个embed_size维度进行卷积操作
    for i in range(x.get_shape()[2]):
        # conv1d要求三维的输入，三维的权重（没有宽度，只有长度。所以称为1d卷积）。具体可以参见官方API。
        # conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])
        conv = tf.nn.tanh(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])
        # [batch_size, sentence_length-ws+1, num_filters_A]
        convs.append(conv)
    # 将embed_size个卷积输出在第三个维度上进行进行stack。所以又获得了一个4位的tensor
    conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_A]
    # 池化。即对第二个维度的sentence_length-ws+1个值取最大、最小、平均值等操作、
    pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_A]
    return pool


def bulid_block_B(x, parameters):
    out = []
    poolings = parameters['poolings']
    filter_size = parameters['filter_size']
    W2 = parameters['W2']
    B2 = parameters['B2']
    with tf.name_scope("bulid_block_B"):
        for pooling in poolings[:-1]:
            pools = []
            for i, ws in enumerate(filter_size[:-1]):
                with tf.name_scope("per_conv-pool-%s" % ws):
                    pool = per_dim_conv_layer(x, W2[i], B2[i], pooling)
                pools.append(pool)
            out.append(pools)
    return out


def similarity_sentence_layer(parameters, X):
    #对输入的两个句子进行构建block_A。
    #sent1,2都是3*3*[batch_size，1， num_filters_A]的嵌套列表
    input_x1 = X[0]
    input_x2 = X[1]
    filter_size = parameters['filter_size']
    num_filters = parameters['filter_num']
    poolings = parameters['poolings']
    sent1 = bulit_block_A(input_x1, parameters)
    sent2 = bulit_block_A(input_x2, parameters)
    fea_h = []
    #实现算法1
    with tf.name_scope("cal_dis_with_alg1"):
        for i in range(3):
            #将max，men，mean三个进行连接
            regM1 = tf.concat(sent1[i], 1)
            regM2 = tf.concat(sent2[i], 1)
            #按照每个维度进行计算max，men，mean三个值的相似度。可以参考图中绿色框
            for k in range(num_filters[0]):
                #comU2计算两个tensor的距离，参见上篇博文，得到一个（batch_size，2）的tensor。2表示余弦距离和L2距离
                fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))
    #得到fea_h是一个长度3*20=60的list。其中每个元素都是（batch_size，2）的tensor
    fea_a = []
    #实现算法2的2-9行
    with tf.name_scope("cal_dis_with_alg2_2-9"):
        for i in range(3):
            for j in range(len(filter_size)):
                for k in range(len(filter_size)):
                    # comU1计算两个tensor的距离，参见上篇博文，上图中的红色框。得到一个（batch_size，3）的tensor。3表示余弦距离和L2距离，L1距离
                    fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))
    #得到fea_a是一个长度为3*3*3=27的list。其中每个元素是（batch_size，3）的tensor

    # 对输入的两个句子进行构建block_B。
    # sent1,2都是2*2*[batch_size，50， num_filters_B]的嵌套列表
    sent1 = bulid_block_B(input_x1, parameters)
    sent2 = bulid_block_B(input_x2, parameters)

    fea_b = []
    # 实现算法2的剩余行
    with tf.name_scope("cal_dis_with_alg2_last"):
        for i in range(len(poolings)-1):
            for j in range(len(filter_size)-1):
                for k in range(num_filters[1]):
                    fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
    ##得到fea_b是一个长度为2*2*20=80的list。其中每个元素是（batch_size，3）的tensor
    return tf.concat(fea_h + fea_a + fea_b, 1)


def create_placeholders(class_num, embedding_dim, fixed_height):
    # X = tf.placeholder(tf.float32, shape=[None, channel_num, None, embedding_dim])  # 样本数，通道数，文本长度， embedding维度
    # Y = tf.placeholder(tf.float32, shape=[None, class_num])
    X = tf.placeholder(tf.float32, shape=[2, None, fixed_height, embedding_dim, 1], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, class_num], name="Y")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return X, Y, keep_prob


def forward_propagation_2(X, keep_prob, parameters, hidden_num, class_num):
    fea = similarity_sentence_layer(parameters, X)
    l1 = tf.contrib.layers.fully_connected(fea, 32)
    l1 = tf.nn.dropout(l1, keep_prob)
    l2 = tf.contrib.layers.fully_connected(l1, 64)
    l2 = tf.nn.dropout(l2, keep_prob)
    l3 = tf.contrib.layers.fully_connected(l2, 32)
    l3 = tf.nn.dropout(l3, keep_prob)
    out = tf.contrib.layers.fully_connected(l3, class_num, activation_fn=None)
    return out


def forward_propagation(X, keep_prob, parameters, hidden_num, class_num):
    fea = similarity_sentence_layer(parameters, X)
    fully_connect_layer_1 = tf.contrib.layers.fully_connected(fea, hidden_num)
    fully_connect_layer_1 = tf.nn.dropout(fully_connect_layer_1, keep_prob)
    out = tf.contrib.layers.fully_connected(fully_connect_layer_1, class_num, activation_fn=None)
    # out = tf.contrib.layers.fully_connected(fea, class_num, activation_fn=None)
    return out


def compute_cost(Z, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y),name="COST")
    all_para = tf.trainable_variables()
    regular = 0.01*tf.reduce_sum([tf.nn.l2_loss(a) for a in all_para])
    cost = cost + regular
    return cost


def similar_model(X_train, Y_train,learning_rate = 0.009,
          num_epochs = 1000, minibatch_size = 64, print_cost = True, save=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    m = X_train.shape[1]
    print("共"+str(m)+"个样本")
    n_y = Y_train.shape[1]
    costs = []
    X, Y, keep_prob = create_placeholders(n_y, 100, 30)
    parameters = initialize_parameters(1, 100)
    Z = forward_propagation(X, keep_prob, parameters, 128, n_y)
    output = tf.nn.softmax(Z, name="output")
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            beg = time.time()
            #  print(str(epoch)+"/"+str(num_epochs))
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            index = 1
            for minibatch in minibatches:
                print("epoch: "+str(epoch)+" batch_num: "+str(index)+"/"+str(num_minibatches+1))
                index += 1
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob:0.5})
                minibatch_cost += temp_cost / num_minibatches

            end = time.time()
            print("epoch "+str(epoch)+" 耗时:"+str((end-beg)//60))
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            costs.append(minibatch_cost)
            save_model(save, epoch, sess)

        if save:
            save_path = "modelFile/model_similar.tf"
            saver = tf.train.Saver()
            saver_path = saver.save(sess, save_path)
            print(save_path + "，已保存")
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 1)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return


def save_model(is_true, epoch, sess):
    if (epoch % 10 == 0) and is_true:
        save_path = "modelFile" + str(epoch) + "/model_similar.tf"
        saver = tf.train.Saver()
        saver_path = saver.save(sess, save_path)
        print(save_path+"，已保存")
    if epoch<10 and epoch>0:
        save_path = "modelFile"+str(epoch)+"/model_similar.tf"
        saver = tf.train.Saver()
        saver_path = saver.save(sess, save_path)
        print(save_path + "，已保存")
    return 0


time_start=time.time() #read train data is a long time process, so we would better record the time so that we can keep everything are under control.
X_train, Y_train = load_dataset("data/train.txt", "model/w2v.model", 30, align_sentence=True)#this step will cost a long time, waiting waiting waiting....
print("#####")
time_end=time.time()
print('totally cost'+str((time_end-time_start)//60)+"mins")
print(X_train.shape)
# 50次收敛
similar_model(X_train, Y_train, 0.0009, num_epochs=100, minibatch_size=512)#this method is responsible for training data
#para1 traindata;para2 traindata label;para3 learning rate;para4 iter num;para5 minibatch size
# 3-layer 32 64 32
