import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from data_process import data_process
from BiLSTM import BiLSTM



def train(file):
    train_data, train_label, vocab_size, output_size = data_process(file)
    # length, depth = np.shape(train_data)
    batch_size = 6
    # print(np.shape(train_data))
    batches = get_batches(train_data, train_label, batch_size)

    # data = np.reshape(train_data, [length//batch_size, batch_size, depth]) # data:(num_batches, B, T)
    # label = np.reshape(train_label, [length//batch_size, batch_size]) # label:(num_batches, B)
        

    max_epochs = 1000

    bilstm = BiLSTM(vocab_size, output_size)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        batches = get_batches(train_data, train_label, batch_size)
        for data, label in batches:
            loss = bilstm.train(sess, data, label)
            # graph = bilstm.save_graph(sess)
            
        print(loss)
        if loss <= 1e-3:
            break
        
    # result = []
    y_pred, y_true = [], []
    test = get_batches(train_data, train_label, batch_size)
    for data, label in test:
        pred = bilstm.predict(sess, data)
        # result.extend(pred-label)
        y_pred.extend(pred)
        y_true.extend(label)
    # result = np.array(result)
    # result[result != 0] = 1
    # accuracy = 1 - np.sum(result)/length
    accuracy = accuracy_score(y_true, y_pred)
    print("the accuracy is %s" %accuracy)
    f1 = f1_score(y_true, y_pred, average='macro')
    print("macro-f1 score is %s" %f1)

    bilstm.save_graph(sess)

def get_batches(words, labels, batch_size):
    idx = [i for i in range(len(words))]
    np.random.shuffle(idx)
    words = words[idx]
    labels = labels[idx]
    length, depth = np.shape(words)
    num_batches = length // batch_size
    last_batch = 0
    if length % batch_size != 0:
        num_batches = num_batches + 1
        last_batch = length % batch_size

    for idx in range(0, length, batch_size):
        x, y = [], []
        if idx < length - last_batch:
            for i in range(batch_size):
                x.extend([words[idx + i]])
                y.extend([labels[idx + i]])
        else:
            for i in range(last_batch):
                x.extend([words[idx + i]])
                y.extend([labels[idx + i]])
        yield x, y
