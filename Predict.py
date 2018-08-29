import tensorflow as tf
from load_file_util import *
import time
#Author:siyouhe666@gmail.com
#this python file can read trained model and use it to predict new data

def Predict(fixed_height,input_path, save_path,model_path, align_sentence=True):
    w2vmodel_path = "model/w2v.model" # This is word2Vec model which you can replace
    begin = time.time()
    labels = []
    data_list = get_data_list(input_path)
    w2vmodel = load_w2vmodel(w2vmodel_path)
    #------------------------
    with tf.Session() as sess:
        meta_path = "modelFile/model_similar.tf.meta" # read the model you have trained
        model_path = "modelFile/" # the same as last step
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        output = graph.get_tensor_by_name("output:0")
        data = []
        sen1_list = []
        sen2_list = []
        for line in data_list:
            no, sen1, sen2 = line
            if align_sentence:
                sen1, sen2 = align_sen(sen1, sen2)
            sen1_matrix = sen2vec(sen1, w2vmodel, fixed_height)
            sen2_matrix = sen2vec(sen2, w2vmodel, fixed_height)
            sen1_matrix = build_matrix(sen1_matrix, 1)
            sen2_matrix = build_matrix(sen2_matrix, 1)
            sen1_list.append(sen1_matrix)
            sen2_list.append(sen2_matrix)
            if len(sen1_list) == 10000:
                data.append(sen1_list)
                data.append(sen2_list)
                data = np.array(data)
                result = sess.run(output, feed_dict={X: data, keep_prob: 1.0})
                label = convert_2_label(result)
                labels += label
                data = []
                sen1_list = []
                sen2_list = []
                print(len(labels))
        if len(sen1_list) >0:
            data.append(sen1_list)
            data.append(sen2_list)
            data = np.array(data)
            result = sess.run(output, feed_dict={X: data, keep_prob: 1.0})
            # result = result*np.array([0.4, 0.6]) # 2
            label = convert_2_label(result)
            labels += label
    print("共"+str(len(labels))+"条结果")
    save_res(labels, save_path)
    end = time.time()
    print("测试完成，耗时： "+str((end-begin)//60)+"min")


def convert_2_label(result):
    res_num = result.shape[0]
    label = []
    for i in range(res_num):
        val = np.argmax(result[i])
        label.append(val)
    return label


# (data_path, w2v_path, fixed_height,align_sentence=False):
"""
def predict(input_path, save_path):
    w2vmodel_path = "model/w2v.model"
    data = load_similar_test_data(input_path, w2vmodel_path, 30)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('modelFile/model_similar.tf.meta')
        saver.restore(sess, tf.train.latest_checkpoint('modelFile/'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        output = graph.get_tensor_by_name("output:0")
        result = sess.run(output, feed_dict={X: data, keep_prob: 1.0})
    label = convert_2_label(result)
    save_res(label, save_path)
    return label
"""


def save_res(label, save_path):
    with open(save_path, 'w', encoding="utf-8") as fout:
        for i in range(len(label)):
            new_line = str(i+1) + "\t" + str(label[i])
            fout.write(new_line)
            fout.write("\n")
    return


Predict(30, "data/test.txt", "result/test_result.txt") #predict new sample you input,para1:the longest length of your sentence,para2: test data path, para3:predict result save path