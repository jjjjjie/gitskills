# -*- coding: utf-8 -*-

import re
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np






class BiLSTM():

    def __init__(self, vocab_size, output_size,
        reuse=None,
        hidd_size=100, 
        embed_size=300,
        learning_r=1e-3,
        scope='BiLSTM'
        ):

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.hidd_size = hidd_size
        self.embed_size = embed_size
        self.learning_r = learning_r
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.graph()

    def graph(self):
        
        # 神经网络的输入和输出。
        self.sent = tf.placeholder(tf.int32, shape=[None, None]) # sent:(B, T)
        self.label = tf.placeholder(tf.int32, shape=[None]) #label:(B)

        # 构建一个查找表。
        self.embed_table = tf.get_variable('embed_table', [self.vocab_size, self.embed_size])
        # x就是句子对应的词向量。
        x = tf.nn.embedding_lookup(self.embed_table, self.sent) #x:(B, T, E)
        
        # 构建正向和反向的LSTM CELL。
        fw_cell = rnn.LSTMCell(self.hidd_size)
        bw_cell = rnn.LSTMCell(self.hidd_size)

        # 通过一个双向rnn。
        _ , final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        final_fw, final_bw = final_state
        ht = tf.concat([final_fw.h, final_bw.h], axis=1) 

        # 经过输出层得到对标签的预测。
        logit = tf.layers.dense(ht, self.output_size) #logit:(B, output_size)
        self.pro = tf.nn.softmax(logit) # pro: (B, output_size)

        #训练部分
        l = tf.one_hot(self.label, self.output_size) # l:(B, output_size)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logit)
        self.cost = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_r).minimize(self.cost)

    def train(self, sess, sent, label):
        loss_p, _ = sess.run([self.cost, self.train_op],{self.sent:sent, self.label:label})
        return loss_p

    def predict(self, sess, sent):
        pro = sess.run(self.pro, {self.sent:sent})
        label_pred = np.argmax(pro, axis=1)
        return label_pred

    def save_graph(self, sess):
        # merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/",sess.graph)  
        # for i in range(100):
        #     result = sess.run(merged,feed_dict={self.sent:sent, self.label:label})
        #     if i%10 == 0:
        #         writer.add_summary(result,i) 
        # return writer



if __name__ == '__main__':
    bilism = BiLSTM(10, 10)
    # bilstm_2 = BiLSTM(10, 10, reuse=True)

    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    
