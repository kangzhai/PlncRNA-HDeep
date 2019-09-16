# PlncRNA-HDeep

import numpy as np
import re
import math
import nltk
import collections
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, normalization, Bidirectional
from keras import optimizers
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

np.random.seed(1337) # random seed

# Evaluation criteria
def comparison(testlabel, resultslabel):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for row1 in range(len(resultslabel)):
        if resultslabel[row1] < 0.5:
            resultslabel[row1] = 0
        else:
            resultslabel[row1] = 1

    for row2 in range(len(testlabel)):

        if testlabel[row2] == 1 and testlabel[row2] == resultslabel[row2]:
            TP = TP + 1
        if testlabel[row2] == 0 and testlabel[row2] != resultslabel[row2]:
            FP = FP + 1
        if testlabel[row2] == 0 and testlabel[row2] == resultslabel[row2]:
            TN = TN + 1
        if testlabel[row2] == 1 and testlabel[row2] != resultslabel[row2]:
            FN = FN + 1

    # TPR：sensitivity, recall, hit rate or true positive rate
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 999999

    # TNR：specificity, selectivity or true negative rate
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 999999

    # PPV：precision or positive predictive value
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 999999

    # NPV：negative predictive value
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 999999

    # FNR：miss rate or false negative rate
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 999999

    # FPR：fall-out or false positive rate
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 999999

    # FDR：false discovery rate
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 999999

    # FOR：false omission rate
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 999999

    # ACC：accuracy
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999

    # F1 score：is the harmonic mean of precision and sensitivity
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999

    # MCC：Matthews correlation coefficient
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 999999

    # BM：Informedness or Bookmaker Informedness
    if TPR != 999999 and TNR != 999999:
        BM = TPR + TNR - 1
    else:
        BM = 999999

    # MK：Markedness
    if PPV != 999999 and NPV != 999999:
        MK = PPV + NPV - 1
    else:
        MK = 999999

    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK


## LSTM using p-nucleotide encoding #####################################################################################################################################
maxlen = 0  # max sequence length
word_freqs = collections.Counter()  # word frequency
num_recs = 0  # number of samples

k = 3 # p-nucleotide encoding
with open('Datasets\\TotalDataset.fasta', 'r+', encoding='gb18030', errors='ignore') as f:
    for line in f:
        name, sentence, label = line.strip().split(",")
        sentence = sentence.lower()
        words=[]
        count=0
        while(count<len(sentence)):
            if(len(sentence[count:count+k])==k):
                words.append(sentence[count:count + k])  # ['tact','actg']
            count += k

        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1

        num_recs += 1

vocab_size = min(64, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(64))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i = 0
with open('Datasets\\TotalDataset.fasta', 'r+', encoding='gb18030', errors='ignore') as f:
    for line in f:
        name, sentence, label = line.strip().split(",")
        sentence = sentence.lower()
        words=[]
        count=0
        while(count<len(sentence)):
            if(len(sentence[count:count+k])==k):
                words.append(sentence[count:count + k])  # ['tact','actg']
            count += k

        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1

X = sequence.pad_sequences(X, maxlen=3100, padding='post')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# Building the model based on LSTM
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=3100))
model.add(Bidirectional(LSTM(64, dropout=0.4)))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Training
model.fit(X_train, y_train, batch_size=128, epochs=3)

# Prediction
loss, accuracy = model.evaluate(X_test, y_test, batch_size=128)
resultslabelLSTM = model.predict(X_test)
## LSTM using p-nucleotide encoding ######################################################################################################################################


## CNN using one-hot encoding ############################################################################################################################################
list = open('Datasets\\TotalDataset.fasta', 'r').readlines()
threshold = 0 # max sequence length
for linelength in list:
    name, sequence, label = linelength.split(',')
    if len(sequence) > threshold:
        threshold = len(sequence)

# one-hot encoding
def onehot(list, threshold):
    onehotsequence = []
    onehotlabel = []
    ATCG = 'ATCG'
    char_to_int = dict((c, j) for j, c in enumerate(ATCG))

    for line in list:
        name, sequence, label = line.split(',')
        rstr = r"[BDEFHIJKLMNOPQRSVWXYZ]"
        sequence = re.sub(rstr, '', sequence)
        integer_encoded = [char_to_int[char] for char in sequence]
        hot_encoded = []
        for value in integer_encoded:
            letter = [0 for _ in range(len(ATCG))]
            letter[value] = 1
            hot_encoded.append(letter)
        # zero-padding
        if len(hot_encoded) < threshold:
            zero = threshold - len(hot_encoded)
            letter = [0 for _ in range(len(ATCG))]
            for i in range(zero):
                hot_encoded.append(letter)

        hot_encoded_array = np.array(hot_encoded).reshape(-1, 4)
        onehotsequence.append(hot_encoded_array)
        onehotlabel.append(label.strip('\n'))

    X = np.array(onehotsequence).reshape(-1, threshold, 4, 1)
    X = X.astype('float32')
    y = np.array(onehotlabel).astype('int').reshape(-1, 1)
    y = np_utils.to_categorical(y, num_classes=2)

    return X, y

X2, y2 = onehot(list, threshold)

traindata, testdata, trainlabel, testlabel = train_test_split(X2, y2, test_size=0.2, random_state=1337)

# Building the CNN model
model = Sequential()
model.add(Convolution2D(batch_input_shape=(None, threshold, 4, 1), filters=32, kernel_size=4, strides=1, padding='same', data_format='channels_last'))
normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=4, strides=4, padding='same', data_format='channels_last'))
model.add(Convolution2D(64, 4, strides=1, padding='same', data_format='channels_first'))
normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
model.add(Activation('relu'))
model.add(MaxPooling2D(4, 4, 'same', data_format='channels_last'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
print('Training --------------')
model.fit(traindata, trainlabel, epochs=10, batch_size=128, verbose=1)

# Prediction
loss, accuracy = model.evaluate(testdata, testlabel, batch_size=128)
resultslabelCNN = model.predict(testdata)
## CNN using one-hot encoding ############################################################################################################################################


## Hybrid deep learning using three strategies ###########################################################################################################################
for rowfuz in range(resultslabelCNN.shape[0]):
    # if abs(2 * resultslabelBiLSTM[rowfuz] - 1) < abs(2 * resultslabelCNN[rowfuz] - 1): # greedy hybrid strategy
    # if abs(2 * resultslabelCNN[rowfuz][0] - 1) >= 0.5: # predominance of CNN hybrid strategy
    if abs(2 * resultslabelLSTM[rowfuz] - 1) < 0.5: # predominance of LSTM hybrid strategy
        resultslabelBiLSTM[rowfuz] = resultslabelCNN[rowfuz]
## Hybrid deep learning using three strategies ###########################################################################################################################


# Evaluation
TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparison(y_test, resultslabelLSTM)


print('PlncRNAPred')
print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
print('ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK)

