import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])
from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

from matplotlib import cm

from tqdm import tqdm

rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams.update({'figure.autolayout': True})

from pprint import pprint

import numpy as np

import pickle

import pandas as pd

import tensorflow as tf

import h5py

# import tflearn

from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Need to pad to this
maxTracks = 20
maxVertices = 10

def lastRelevant(output, length):

    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb

    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def lastRelevantBiDir(output, length):

    # For bidirectional RNN concatenate forward and backward outputs
    # (Check this is correct wrt padding)

    batch_size = tf.shape(output[0])[0]
    max_length = tf.shape(output[0])[1]
    out_size = int(output[0].get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flatFwd = tf.reshape(output[0], [-1, out_size])
    flatBwd = tf.reshape(output[1], [-1, out_size])
    relevantFwd = tf.gather(flatFwd, index)
    relevantBwd = tf.gather(flatBwd, index)
    relevant = tf.concat((relevantFwd, relevantBwd), 1)
    return relevant

def gruCell(n_hidden):
    return tf.contrib.rnn.GRUCell(n_hidden, kernel_initializer = tf.contrib.layers.xavier_initializer())

def lstmCell(n_hidden, training):
    # return BNLSTMCell(n_hidden, training, max_bn_steps = 100)
    return tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden)

def dynamicRNN(x, seqlen, training, n_hidden, name):

    with tf.name_scope(name) as scope:

        keepProb = tf.cond(training[0][0], lambda : 0.75, lambda : 1.0)

        gru = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(gruCell(n_hidden), output_keep_prob = keepProb, input_keep_prob = keepProb) for g in range(10)])

        output, states = tf.nn.dynamic_rnn(gru, inputs = x, dtype = tf.float32, sequence_length = seqlen, scope = scope)

        last = lastRelevant(output, seqlen)

        return last

def NN(t, v, b, seqlenTracks, seqlenVertices, training, n_hidden, tMask, vMask):

    # combRNN = tf.contrib.layers.flatten(t) # Remove RNN and just use flattened input data from nFeatures * nMaxTracks

    tShape = t.get_shape().as_list()
    vShape = v.get_shape().as_list()

    inputTracksOnly = tf.reshape(t, [-1, tShape[2]])
    inputVerticesOnly = tf.reshape(v, [-1, vShape[2]])

    lt1 = tf.layers.dense(inputTracksOnly, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    dt1 = tf.layers.dropout(lt1, 0.5, training = training[0][0])
    lt2 = tf.layers.dense(dt1, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    # dt2 = tf.layers.dropout(lt2, 0.5, training = training[0][0])
    dt2 = tf.layers.batch_normalization(lt2, training = training[0][0])
    lt3 = tf.layers.dense(dt2, 20, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())

    lv1 = tf.layers.dense(inputVerticesOnly, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    dv1 = tf.layers.dropout(lv1, 0.5, training = training[0][0])
    lv2 = tf.layers.dense(dv1, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    # dv2 = tf.layers.dropout(lv2, 0.5, training = training[0][0])
    dv2 = tf.layers.batch_normalization(lv2, training = training[0][0])
    lv3 = tf.layers.dense(dv2, 20, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())

    rnnInputTracks = tf.reshape(lt3, [-1, tShape[1], 20])
    rnnInputVertices = tf.reshape(lv3, [-1, vShape[1], 20])

    trackRNN = dynamicRNN(rnnInputTracks, seqlenTracks, training, n_hidden, 'trackRNN') * tMask
    vertexRNN = dynamicRNN(rnnInputVertices, seqlenVertices, training, n_hidden, 'vertexRNN') * vMask

    lb1 = tf.layers.dense(b, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    db1 = tf.layers.dropout(lb1, 0.5, training = training[0][0])
    lb2 = tf.layers.dense(db1, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    # db2 = tf.layers.dropout(lb2, 0.5, training = training[0][0])
    db2 = tf.layers.batch_normalization(lb2, training = training[0][0])
    lb3 = tf.layers.dense(db2, 20, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())

    # bOut = 1?

    combRNN = tf.concat((trackRNN, vertexRNN, lb3), 1)

    l1 = tf.layers.dense(combRNN, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    d1 = tf.layers.dropout(l1, 0.5, training = training[0][0])
    l2 = tf.layers.dense(d1, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    d2 = tf.layers.dropout(l2, 0.5, training = training[0][0])
    l3 = tf.layers.dense(d2, n_hidden, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer())
    out = tf.layers.dense(l3, 1, activation = None, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'trainOut')

    return (out, tf.nn.sigmoid(out, name = 'predOut'))

tracksFile = h5py.File('tracksTFNew.h5', 'r')
tracks = np.array(tracksFile.get('tracks'))

verticesFile = h5py.File('verticesTFNew.h5', 'r')
vertices = np.array(verticesFile.get('vertices'))

dataB = pd.read_hdf('bDataPDNew.h5')

dataB['charge'] = dataB['charge'].replace(-1, 0)

# For those with 0 tracks/vertices, have a dummy entry which is zero

zeroTracksMask = np.array(list(map(lambda x : 0 if x < 1 else 1, dataB['nTracks'].as_matrix()))).reshape(-1, 1)
zeroVerticesMask = np.array(list(map(lambda x : 0 if x < 1 else 1, dataB['nVertices'].as_matrix()))).reshape(-1, 1)

dataB['zeroTracksMask'] = zeroTracksMask
dataB['zeroVerticesMask'] = zeroVerticesMask

ids = set(dataB['evtNumber'])

seqlensData = np.zeros(len(ids))

nTrackFeatures = tracks.shape[2]
nVertexFeatures = vertices.shape[2]

scalerTracks = StandardScaler()
scalerVertices = StandardScaler()

tracks = scalerTracks.fit_transform(tracks.reshape(-1, tracks.shape[1] * tracks.shape[2])).reshape(-1, maxTracks, nTrackFeatures)
vertices = scalerVertices.fit_transform(vertices.reshape(-1, vertices.shape[1] * vertices.shape[2])).reshape(-1, maxVertices, nVertexFeatures)

dataIndices = range(len(tracks))

trainDataIndices, testDataIndices, trainLabel, testLabel = train_test_split(dataIndices, dataB, test_size = 0.05, random_state = 42, stratify = dataB['charge'])

trainDataTracks = tracks[trainDataIndices]
trainDataVertices = vertices[trainDataIndices]

testDataTracks = tracks[testDataIndices]
testDataVertices = vertices[testDataIndices]

seqlensTracksTrain = trainLabel['nTracks'].replace(0, 1).as_matrix() # Use dummy single entry if zero, but mask as above
seqlensTracksTest = testLabel['nTracks'].replace(0, 1).as_matrix()

seqlensVerticesTrain = trainLabel['nVertices'].replace(0, 1).as_matrix()
seqlensVerticesTest = testLabel['nVertices'].replace(0, 1).as_matrix()

zeroTracksMaskTrain = trainLabel['zeroTracksMask'].as_matrix().reshape(-1, 1)
zeroTracksMaskTest = testLabel['zeroTracksMask'].as_matrix().reshape(-1, 1)

zeroVerticesMaskTrain = trainLabel['zeroVerticesMask'].as_matrix().reshape(-1, 1)
zeroVerticesMaskTest = testLabel['zeroVerticesMask'].as_matrix().reshape(-1, 1)

trainB = trainLabel.drop('charge', 1).drop('nSelectedTracks', 1).drop('nSelectedVertices', 1).drop('nTracks', 1).drop('nVertices', 1).drop('evtNumber', 1)
testB = testLabel.drop('charge', 1).drop('nSelectedTracks', 1).drop('nSelectedVertices', 1).drop('nTracks', 1).drop('nVertices', 1).drop('evtNumber', 1)

nBFeatures = len(list(trainB))

scalerB = StandardScaler()

trainB = scalerB.fit_transform(trainB.as_matrix())
testB = scalerB.transform(testB.as_matrix())

trainLabel = trainLabel['charge'].as_matrix().reshape(-1, 1)
testLabel = testLabel['charge'].as_matrix().reshape(-1, 1)

n_hidden = 64
n_classes = 2
learning_rate = 0.001

# Placeholders -> name so that they can be recovered form a saved model

t = tf.placeholder("float", [None, maxTracks, nTrackFeatures], name = 't')
v = tf.placeholder("float", [None, maxVertices, nVertexFeatures], name = 'v')
y = tf.placeholder("float", [None, n_classes - 1]), name = 'y'
b = tf.placeholder("float", [None, nBFeatures], name = 'b')

training = tf.placeholder("bool", [None, 1], name = 'training')

# A placeholder for indicating each sequence length
seqlenTracks = tf.placeholder(tf.int32, [None], name = 'seqlenTracks')
seqlenVertices = tf.placeholder(tf.int32, [None], name = 'seqlenVertices')

tMask = tf.placeholder("float", [None, 1], name = 'tMask')
vMask = tf.placeholder("float", [None, 1], name = 'vMask')

pred = NN(t, v, b, seqlenTracks, seqlenVertices, training, n_hidden, tMask, vMask)

# Define loss and optimizer -> logit output for this
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred[0], labels=y))

optimizer = None
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step (important for batch norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_steps = 10000
batchSize = 1000

loss = []
lossTest = []
rocTrain = []
rocTest = []

test_dataTracks, test_dataVertices, test_label, test_seqlenTracks, test_seqlenVertices = testDataTracks[:1000], \
                                                                                         testDataVertices[:1000], \
                                                                                         testLabel[:1000], \
                                                                                         seqlensTracksTest[:1000], \
                                                                                         seqlensVerticesTest[:1000]
test_dataTMask, test_dataVMask = zeroTracksMaskTest[:1000], \
                                 zeroVerticesMaskTest[:1000]
test_B = testB[:1000]

# To configure the number of CPU threads

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=8,
      inter_op_parallelism_threads=8)

with tf.Session() as sess:
# with tf.Session(config=session_conf) as sess:

    sess.run(init)

    batchStep = 0

    for step in tqdm(range(0, training_steps)):

        # Get the mini-batch -> probably slow
        if (batchStep + 1) * batchSize > len(trainDataTracks):
            batchStep = 0

            # shuffle our training data using index array perm

            perm = np.random.permutation(range(trainDataTracks.shape[0]))

            trainDataTracks = trainDataTracks[perm]
            trainDataVertices = trainDataVertices[perm]

            trainLabel = trainLabel[perm]

            seqlensTracksTrain = seqlensTracksTrain[perm]
            seqlensVerticesTrain = seqlensVerticesTrain[perm]

            zeroTracksMaskTrain = zeroTracksMaskTrain[perm]
            zeroVerticesMaskTrain = zeroVerticesMaskTrain[perm]

            trainB = trainB[perm]

        # Select the mini-batches

        batch_t, batch_v, batch_y, batch_seqlenTracks, batch_seqlenVertices = trainDataTracks[batchStep * batchSize : batchStep * batchSize + batchSize], \
                                                                              trainDataVertices[batchStep * batchSize : batchStep * batchSize + batchSize], \
                                                                              trainLabel[batchStep * batchSize : batchStep * batchSize + batchSize], \
                                                                              seqlensTracksTrain[batchStep * batchSize : batchStep * batchSize + batchSize], \
                                                                              seqlensVerticesTrain[batchStep * batchSize : batchStep * batchSize + batchSize]

        batch_trackMask, batch_vertexMask = zeroTracksMaskTrain[batchStep * batchSize : batchStep * batchSize + batchSize], \
                                            zeroVerticesMaskTrain[batchStep * batchSize : batchStep * batchSize + batchSize]

        batch_b = trainB[batchStep * batchSize : batchStep * batchSize + batchSize]

        # Run the optimisation

        _, l = sess.run([optimizer, cost], feed_dict={t: batch_t, v: batch_v, b : batch_b, y: batch_y, seqlenTracks: batch_seqlenTracks, seqlenVertices: batch_seqlenVertices, tMask : batch_trackMask, vMask : batch_vertexMask, training : np.array([True]).reshape(-1, 1)})
        loss.append(l)

        # Calculate the loss on test data

        l = sess.run([cost], feed_dict={t: test_dataTracks, v: test_dataVertices, b : test_B, y: test_label, seqlenTracks: test_seqlenTracks, seqlenVertices: test_seqlenVertices, tMask : test_dataTMask, vMask : test_dataVMask, training : np.array([False]).reshape(-1, 1)})[0]
        lossTest.append(l)

        batchStep += 1

        if step % 100 == 0:

            # Save the ROC AUC at this step
            # Only use a subset to speed things up

            predTest = sess.run(pred[1], feed_dict={t: testDataTracks[:10000], v: testDataVertices[:10000], b : testB[:10000], y: testLabel[:10000], seqlenTracks: seqlensTracksTest[:10000], seqlenVertices: seqlensVerticesTest[:10000], tMask : zeroTracksMaskTest[:10000], vMask : zeroVerticesMaskTest[:10000], training : np.array([False]).reshape(-1, 1)})
            predTrain = sess.run(pred[1], feed_dict={t: trainDataTracks[:10000], v: trainDataVertices[:10000], b : trainB[:10000], y: trainLabel[:10000], seqlenTracks: seqlensTracksTrain[:10000], seqlenVertices: seqlensVerticesTrain[:10000], tMask : zeroTracksMaskTrain[:10000], vMask : zeroVerticesMaskTrain[:10000], training : np.array([False]).reshape(-1, 1)})
            rocaucTrain = roc_auc_score(trainLabel[:10000], predTrain.reshape(-1, 1))
            rocaucTest = roc_auc_score(testLabel[:10000], predTest.reshape(-1, 1))
            rocTrain.append(rocaucTrain)
            rocTest.append(rocaucTest)

    print("Optimization Finished!")

    predTest = sess.run(pred[1], feed_dict={t: testDataTracks, v: testDataVertices, b : testB, y: testLabel, seqlenTracks: seqlensTracksTest, seqlenVertices: seqlensVerticesTest, tMask : zeroTracksMaskTest, vMask : zeroVerticesMaskTest, training : np.array([False]).reshape(-1, 1)})
    predTrain = sess.run(pred[1], feed_dict={t: trainDataTracks, v: trainDataVertices, b : trainB, y: trainLabel, seqlenTracks: seqlensTracksTrain, seqlenVertices: seqlensVerticesTrain, tMask : zeroTracksMaskTrain, vMask : zeroVerticesMaskTrain, training : np.array([False]).reshape(-1, 1)})

    rocaucTrain = roc_auc_score(trainLabel, predTrain.reshape(-1, 1))
    rocaucTest = roc_auc_score(testLabel, predTest.reshape(-1, 1))

    print('ROC AUC TRAIN:', rocaucTrain)
    print('ROC AUC TEST:', rocaucTest)

    fpr, tpr, thresholds = roc_curve(testLabel, predTest.reshape(-1, 1))

    plt.plot(1. - fpr, tpr, lw = 1)
    plt.savefig('ftagROCNew.pdf')
    plt.clf()

    # Save the model

    saver = tf.train.Saver()
    saver.save(sess, 'ftagRNN')

plt.plot(loss, lw = 1.0)
plt.plot(lossTest, lw = 1.0)
plt.savefig('ftagLossNew.png')
plt.clf()

plt.plot(rocTrain, lw = 1.0)
plt.plot(rocTest, lw = 1.0)
plt.savefig('ftagROCNew.png')
