# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:30:25 2017

@author: admin
"""

import numpy as np
import pandas as pnds

from matplotlib import pyplot as pp
from scipy.io import wavfile
from scipy.signal import lfilter, firls, decimate, firwin, butter, resample

def detect_beats(signal, fs, mean_fs, cutoff_freq, variance_buffer):
    signal = get_low_pass_filtered(np.abs(signal), fs, cutoff_freq)
    decimated = resample_signal(signal, fs, mean_fs)
    variance = get_variance(decimated, n_taps=43)
    mean_power = get_mean_power(decimated, n_taps=43)
    C = -0.0000015*variance+1.51
    thresh = C*mean_power
    return decimated > thresh

def get_low_pass_filtered(signal, fs, cutoff_freq):
    b = firwin(numtaps=777, cutoff=cutoff_freq/(fs/2.))
    a = [1.]
    ret = lfilter(b, a, signal)
    return ret

def get_mean_power(signal, n_taps=43):
    buffer = [0. for i in range(n_taps)]
    out = []
    for el in signal:
        buffer.pop(0)
        buffer.append(el)
        variance = [np.mean(buffer) for buff_el in buffer]
        out.append(np.mean(variance))
        
    return np.array(out)
    
def resample_signal(signal, fs, req_fs):
    return resample(signal, num=int(len(signal)*(req_fs/fs)))
    
def decimate_signal(signal, fs, req_fs):
    decimate_factor = fs/req_fs
    return decimate(signal, int(decimate_factor))

def get_variance(signal, n_taps=43):
    buffer = [0. for i in range(n_taps)]
    out = []
    for el in signal:
        buffer.pop(0)
        buffer.append(el)
        buff_mean = np.mean(buffer)
        variance = [(buff_mean-buff_el)**2 for buff_el in buffer]
        out.append(np.mean(variance))
        
    return np.array(out)

def detect_edges_beats(beats):
    prev = beats[0:-1:]
    prev = np.append(np.array([False]), prev)
    
    xor = np.logical_xor(prev, beats)
    beats = np.logical_and(beats, xor)
    return beats

def beat_intervals(beat_detections, fs):
    out = []
    dt = 1./fs
    prev = None
    for i, el in enumerate(beat_detections):
        if el and (prev != None):
            time = i*dt
            bpm = 60*(1./(time-prev))
            while not (bpm > 80. and bpm < 180):
                bpm /= 2.
            out.append(bpm)
            prev = time
        elif el:
            prev = i*dt
        
    return np.array(out)

def bpm_adaptive_filter(beat_signal, buff_size):
    pass

mean_freq = 4000.
file_to_test = r"C:\Github\BeatDetector\BeatDetector\Music\140bpm_2.wav"
fs, raw_data = wavfile.read(file_to_test)

data = None
if len(raw_data.shape) > 1:
    data = raw_data[:, 0]
else:
    data = raw_data[:]
    
data = data.astype('float')/np.max(data)

beats = detect_beats(data, fs, mean_freq, 7., 43)
beats = detect_edges_beats(beats)

beat_intervals = beat_intervals(beats, mean_freq)

beats_x = np.linspace(0., 1., beats.shape[0])
data_x = np.linspace(0., 1., data.shape[0])

pp.figure()
pp.plot(data_x, np.abs(data))
pp.plot(beats_x, beats)

print(np.mean(beat_intervals))

#pp.figure()
#pp.plot(x_data, data)
#pp.plot(beat_x, beats)




    
