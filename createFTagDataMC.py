# B

# charge
# PT
# B_DIRA_OWNPV
# B_IP_OWNPV
# eta
# phi
# B_VD_OWNPV
# B_OWNPV_NTRACKS
# nSelectedTracks
# nSelectedVertices

# tracks

# pide
# pidmu
# pidk
# pidpi
# pidp
# ismuon
# P
# PT
# charge
# track_OWNPV_CHI2NDOF - consistency with own pv
# track_BPV_IP_OWNPV - distance from B PV
# track_BPV_IPCHI2_OWNPV - consistency with B pv
# trghostprob
# iTrack
# runNumber
# evtNumber
# inVertex

# vertex

# iVertex
# runNumber
# evtNumber
# vertex_IP_OWNPV - ip wrt own PV
# vertex_BPV_IP_OWNPV - ip wrt B PV
# vertex_OWNPV_CHI2NDOF - consistency with own pv
# mass
# PT
# P
# ctau


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

import h5py

rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams.update({'figure.autolayout': True})

from pprint import pprint

import numpy as np

import pickle

import pandas as pd

from root_pandas import read_root

bVars = [

'evtNumber',
'runNumber',
'charge',
'PT',
'B_DIRA_OWNPV',
'B_IP_OWNPV',
'eta',
'phi',
'B_VD_OWNPV',
'B_OWNPV_NTRACKS',
'nSelectedTracks',
'nSelectedVertices',

]

trackVars = [

'evtNumber',
'runNumber',
'pide',
'pidmu',
'pidk',
'pidpi',
'pidp',
'ismuon',
'P',
'PT',
'charge',
'track_OWNPV_CHI2NDOF',
'track_BPV_IP_OWNPV',
'track_BPV_IPCHI2_OWNPV',
'trghostprob',
# 'iTrack',
# 'inVertex',

]

vertexVars = [

'evtNumber',
'runNumber',
# 'iVertex',
'vertex_IP_OWNPV',
'vertex_BPV_IP_OWNPV',
'vertex_OWNPV_CHI2NDOF',
'mass',
'PT',
'P',
'ctau',

]

maxTracks = 20
maxVertices = 10

bData = read_root('/data/lhcb/phrnas/ftag/B2JPsiK-TagInfo-Combined-New.root', 'TagInfo/B', columns = bVars)
trackData = read_root('/data/lhcb/phrnas/ftag/B2JPsiK-TagInfo-Combined-New.root', 'TagInfo/tracks', columns = trackVars)
vertexData = read_root('/data/lhcb/phrnas/ftag/B2JPsiK-TagInfo-Combined-New.root', 'TagInfo/vertices', columns = vertexVars)

plt.hist(bData.query('nSelectedTracks < 50')['nSelectedTracks'], bins = 50)
plt.savefig('nSelTracks.pdf')
plt.clf()

plt.hist(bData.query('nSelectedVertices < 30')['nSelectedVertices'], bins = 30)
plt.savefig('nSelVertices.pdf')
plt.clf()

# trackData = trackData.drop('iTrack', 1).drop('inVertex', 1)
# vertexData = vertexData.drop('iVertex', 1)

# To 'make' evtNumber unique wrt multiple runs in MC!

bData['evtNumber'] = (bData['evtNumber'].apply(str) + bData['runNumber'].apply(str)).apply(int)
trackData['evtNumber'] = (trackData['evtNumber'].apply(str) + trackData['runNumber'].apply(str)).apply(int)
vertexData['evtNumber'] = (vertexData['evtNumber'].apply(str) + vertexData['runNumber'].apply(str)).apply(int)

bData = bData.drop('runNumber', 1)
trackData = trackData.drop('runNumber', 1)
vertexData = vertexData.drop('runNumber', 1)

ids = set(bData['evtNumber'])

nTrackFeatures = len(list(trackData)) - 1 # no ID
nVertexFeatures = len(list(vertexData)) - 1 # no ID

tracks = np.zeros((len(ids), maxTracks, nTrackFeatures))
vertices = np.zeros((len(ids), maxVertices, nVertexFeatures))

print(tracks.shape, trackData.as_matrix().shape)

# TODO: Check what 'nan' value takes in the new data

# Insert dummy zero entries for events with no tracks/vertices

zeroTracksEvts = bData.query('nSelectedTracks == 0')['evtNumber'].as_matrix()
zeroVerticesEvts = bData.query('nSelectedVertices == 0')['evtNumber'].as_matrix()

zeroTrackRows = np.zeros((len(zeroTracksEvts), len(list(trackData))))
zeroVertexRows = np.zeros((len(zeroVerticesEvts), len(list(vertexData))))

for i in tqdm(range(len(zeroTracksEvts))):
    zeroTrackRows[i][0] = zeroTracksEvts[i]

for i in tqdm(range(len(zeroVerticesEvts))):
    zeroVertexRows[i][0] = zeroVerticesEvts[i]

trackData = trackData.append(pd.DataFrame(zeroTrackRows, columns = list(trackData)))
vertexData = vertexData.append(pd.DataFrame(zeroVertexRows, columns = list(vertexData)))

# Sort by event number (to match order with B data), group by event number, and then sort by IP
# Slower than sorting once by IP, but easier to match with B

sortedTracks = trackData.sort_values('evtNumber', ascending = True)
sortedVertices = vertexData.sort_values('evtNumber', ascending = True)

bData = bData.sort_values('evtNumber', ascending = True)

i = 0
for e, group in tqdm(sortedTracks.groupby('evtNumber', sort = False)): # sort = True by default

    t = group.sort_values('track_BPV_IP_OWNPV', ascending = True).drop('evtNumber', 1).as_matrix()[:maxTracks, :]

    # Insert data into padded array via slice
    tracks[i,:t.shape[0], :t.shape[1]] = t

    i += 1

i = 0
for e, group in tqdm(sortedVertices.groupby('evtNumber', sort = False)): # sort = True by default

    v = group.sort_values('vertex_BPV_IP_OWNPV', ascending = True).drop('evtNumber', 1).as_matrix()[:maxVertices, :]

    vertices[i,:v.shape[0], :v.shape[1]] = v

    i += 1

bData['nTracks'] = np.minimum(bData['nSelectedTracks'].as_matrix(), maxTracks)
bData['nVertices'] = np.minimum(bData['nSelectedVertices'].as_matrix(), maxVertices)

hf = h5py.File('/data/lhcb/phrnas/ftag/tracksTFNew.h5', 'w')
hf.create_dataset('tracks', data=tracks)
hf.close()

hf = h5py.File('/data/lhcb/phrnas/ftag/verticesTFNew.h5', 'w')
hf.create_dataset('vertices', data=vertices)
hf.close()

bData.to_hdf('/data/lhcb/phrnas/ftag/bDataPDNew.h5', 'B', mode = 'w')
trackData.to_hdf('/data/lhcb/phrnas/ftag/trackDataPDNew.h5', 'Tracks', mode = 'w')
vertexData.to_hdf('/data/lhcb/phrnas/ftag/vertexDataPDNew.h5', 'Vertices', mode = 'w')
