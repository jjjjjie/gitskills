import numpy as np


def data_process(file):
    with open(file) as f:
        text = f.read()
    text = text.replace('？','')
    text = text.replace('?','')
    text = text.replace('"','')
    text = text.replace('！','')
    text = text.replace('，','')
    text = text.replace('、','')
    text = text.replace('-','')

    words = text.split()

    inp = words[::2]
    out = words[1::2]
    vocab = set([])
    label = set([])
    for i in range(len(inp)):
        vocab = vocab | set(inp[i])
    len_vocab = len(vocab)


    label = set(out)
    len_label = len(label)

    vocab_to_int = {w: c for c, w in enumerate(vocab)}
    label_to_int = {w: c for c, w in enumerate(label)}
    length = np.zeros(len(inp))
    for i in range(len(inp)):
        length[i] = len(inp[i])
    max_len = np.int(max(length))


    num_words = np.zeros([len(inp),max_len])
    num_labels = np.zeros(len(out))
    for i in range(len(inp)):
        for j in range(len(inp[i])):
            num_words[i][j] = vocab_to_int[inp[i][j]]

    for i in range(len(out)):
        num_labels[i] = label_to_int[out[i]]

    return num_words, num_labels, len_vocab, len_label

if __name__ == '__main__':
    x, y, z, a= data_process('/home/liujie/project/corpus.txt')
    print(y)
    print(z)