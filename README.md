B flavour tagging with recurrent neural networks (TensorFlow)
===

This repository comprises a very rough first attack on B flavour tagging with a recurrent neural network, implemented in [TensorFlow](https://www.tensorflow.org/).

The files here represent the three stages to go from the input DST files to the resulting (uncalibrated) classifier:

* `benderTracks.py`: Bender script to extract simulated track, vertex and signal B+ -> J/Psi K+ information.
* `createFTagDataMC.py`: Pre-process ROOT files into the format necessary for the neural network, and output HDF5 files.
* `ftagRNN.py`: Train and perform a very quick evaluation of a neural network in TensorFlow to classify the B flavour.

Motivation
---

Recurrent neural networks are networks that are specialised to process variable length *sequences* of data, and consist of nodes with an input, output, and connection to the next node in the sequence. These nodes also share a common set of weights, which in comparison to conventional feedforward networks, permit location invariance whilst remaining sensitive to information propagated directionally through the network.

Recurrent neural networks can therefore be used to extract information from track-based observables, where correlations between tracks are important, but the overall number or configuration of tracks within a sequence is less important.

In comparison to convolutional neural networks, where convolutions are performed over restricted regions of an input dataset, recurrent neural networks in principle allow propagation of information over very large distances. As such, they are often used for machine translation, image captioning, and voice recognition.

Data
---

The dataset input to `benderTracks.py` is assumed to be Stripping20 B+ -> J/Psi(mumu) K+ MC. For each event with a signal B+ decay, variables related to each track and vertex in the event are stored, along with variables corresponding to the B+ decay. These variables are listed below, and are not intended to be an optimal set (or even a superset of the optimal set).

Tracks are obtained from

* `StdNoPIDsPions`
* `StdAllLooseElectrons`
* `StdAllLooseMuons`

where, if two or more tracks are present in the same container, only one is saved (hopefully without any loss of information).

Vertex information is obtained from particles constructed from two (or more) tracks, from those particles present in

* `StdLooseDetachedDipion`
* `StdLooseDiMuon`
* `StdLooseD02KPi`

where, if two or more protoparticles corresponding to the particles in those containers overlap, then only the particle with the best vertex fit chi2 is saved (this greedy allocation does not result in a globally minimal vertex chi2, but hopefully is sufficient). These containers are *not* chosen to optimise the tagging information, but rather in an effort to obtain a reasonable number of vertices (and are very much not an optimal choice).

Tracks or vertices (or more specifically, any protoparticles) present in the B+ signal decay are removed.

Data pre-preprocessing
---

* Track related inputs to RNN layers are sorted in order of ascending track impact parameter with respect to the signal primary vertex. Vertex related inputs to RNN layers are sorted in order of ascending impact parameter of the combination of the tracks that form the vertex, with respect to the signal primary vertex.

* Events with zero tracks or vertices present are still input into the classifier. In these cases, a single dummy entry to the track or vertex vector is present, and during training the output of the corresponding RNN for this event is masked (multiplied by zero). Currently events with zero tracks and zero vertices are present, however in future it would be better to remove those events with zero tracks (as very little tag information will come from the signal B information).

* Whilst the RNN architecture accepts variable length input for the number of time-steps, the Tensorflow API requires that all rows of the input vector are the same shape. Hence, the time-step axis of the input tensor is padded to the length of the longest number of time-steps. In this case, a maximum of 20 tracks and 10 vertices are considered.

* A network with arbitrary input-scales results in an elliptical error surface, which inhibits the convergence of the gradient descent optimisation methods. To improve convergence, inputs are scaled such that each input variable forms a unit Gaussian by subtracting the mean and dividing by the standard deviation of the distribution. This is performed using the `scikit-learn` class `sklearn.preprocessing.StandardScaler`, separately for track, vertex, and signal B quantities.

* The classification classes, corresponding to a B+ or B− signal decay, are encoded such that the output for B− , is 0, and the output for B+ is 1.

* The total data sample is split into a training set and a test set using the `scikit-learn` function `sklearn.cross validation.train_test_split`, with 5% of the data being used for testing purposes. The split is stratiﬁed such that there is equal amounts of each class (B+ and B− ) in each of the training and test sets.

Network configuration
---

* The activation functions of the feedforward fully-connected (dense) layers are the rectified linear unit (ReLU), f(x) = max(0, x). The ReLU activation function is fast to compute, and does not have the problem common to tanh or sigmoid activation functions where the gradient can vanish during back-propagation.

* The weights of fully-connected layers, W, for layers with ReLU activations, are randomly initialised from a Gaussian distribution with zero mean, and variance equal to Var(W) = 2/n_in, where n_in is equal to the number of inputs to the layer. This aids convergence of the network by ensuring that forward and backward propagated values through the layer do not rapidly vanish or explode.

* In the recurrent network, gated recurrent units (GRU) are used, which are simplified versions of long short-term memory (LSTM) units that lack an output gate, and are more efficient for equivalent performance.

* Overfitting is reduced by randomly setting the output of a proportion, p, of neurons in fully-connected layers, to zero, each iteration whilst training. This ad-hoc regularisation technique results in the network being forced to learn more distributed representations of the data (and therefore the final network can be thought of as the average over all networks with a specific configuration of active and inactive neurons). In all cases, p is set to 0.5.

* Layer inputs to the RNN are 'batch' normalised to mitigate the detrimental effect of saturation of the non-linear activation function when performing back-propagation. This transformation is calculated and applied separately for each mini-batch, and allows training of deeper networks more efficiently and with better robustness to the initialisation conditions. As the input to the RNN also depends on the other inputs within the mini-batch, this also acts as a form of regularisation to reduce overtraining. No batch-normalisation is performed within the RNN itself.

* Optimisation is performed using the Adam optimiser. The loss function used is the mean of the cross-entropy, as implemented in the TensorFlow function `tf.nn.sigmoid_cross_entropy_with_logits`. For computational efficiency, this incorporates a sigmoid output transformation internally (as such the network has a 'raw' logit output that is used for the optimiser, and a second sigmoid output that is used for output predictions).

Network structure
---

* Track/vertex/signal B variables are pre-processed as flattened arrays of dimension (nData, nTrackFeatures) by a 4 layer feedforward network.

* The output of this track and vertex networks are batch-normalised, reshaped to re-associate tracks/vertices to events (to dimension (nData, nTracks, nTrackFeatures)), and fed into a 10 layer recurrent network, for tracks and vertices separately.

* The last relevant output is extracted using the number of tracks/vertices in each event (to account for the padding), which is concatenated with the output from the other RNN and the output of the signal B feedforward network. This is passed through a 4 layer feedfoward network, the output of which is used for the loss function and tag prediction.

Future improvements
---

* Train and evaluate with more (real) data
* Optimise input variables
* Investigate (batch) normalisation within recurrent network
* Optimise hyperparameters and layer configuration (number/width)
* Bi-directional RNNs
* Highway/residual networks and skip connections - improve training of deeper networks by using 'shortcuts' (particularly hidden -> hidden skips for RNNs)
* Incorporate better track <-> vertex relationship in the network structure: grid LSTMs (exactly what it sounds like) and recursive neural networks (at each node have a further recurrent network that represents the vertex -> track hierarchy)


Hyperparameters
---

Some unoptimised hyperparameters that perform okay:

* TensorFlow version: v1.4.1
* Training steps: 10000
* Mini-batch size: 1000
* Adam learning rate: 0.001
* Number of hidden units: 64
* Number of RNN layers: 10
* Number of feedforward layers (pre-processing): 3
* Number of feedforward layers (post-processing): 4

Files
---

* Stripping20 B+ -> J/Psi(mumu) K+ 2012 MC, 12143001, 2.5M events

* Bender v30r6 output:
`/lhcb/user/d/dohanlon/GangaFiles_15.08_Wednesday_24_January_2018/B2JPsiK-TagInfo-Combined-New.root`
* Track data (20 tracks per event):  `/lhcb/user/d/dohanlon/GangaFiles_15.00_Wednesday_24_January_2018/trackDataPDNew.h5`
* Vertex data (10 vertices per event): `/lhcb/user/d/dohanlon/GangaFiles_15.03_Wednesday_24_January_2018/vertexDataPDNew.h5`
* B data: `/lhcb/user/d/dohanlon/GangaFiles_14.46_Wednesday_24_January_2018/bDataPDNew.h5`


References
---

* [Deep neural networks](http://www.deeplearningbook.org/)
* [The unreasonable effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs)
* [Glorot initialisation](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [Dropout](https://arxiv.org/abs/1207.0580)
* [Batch normalisation](https://arxiv.org/abs/1502.03167)

* [*Recursive* neural networks](https://arxiv.org/abs/1702.00748)
* [Inclusive flavour tagging](https://arxiv.org/abs/1705.08707)
* [ATLAS b jet tagging](http://cds.cern.ch/search?ln=en&p=ATL-PHYS-PUB-2017-003)
