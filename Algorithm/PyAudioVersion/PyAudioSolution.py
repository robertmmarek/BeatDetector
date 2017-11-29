# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:13:35 2017

@author: admin
"""
import struct
import queue
import time
import math

import pyaudio as pa
import numpy as np
import multiprocessing as mp


from matplotlib import pyplot as pp
from multiprocessing import Queue, Process, Pipe, Value
from scipy.fftpack import fft


class AcquisitionSettings:
    def __init__(self,
                 frame_length=128,
                 fs=5000):
        self.frame_length = frame_length
        self.fs = fs
        
def acquisition_loop(acq_settings, output_queue, stop_value):
    p = pa.PyAudio()
    mic = p.open(format=pa.paFloat32,
                 rate=acq_settings.fs,
                 channels=True,
                 input=True)
    
    mic.start_stream()
    
    while not stop_value.value:
        raw_data = mic.read(acq_settings.frame_length)
        float_data = struct.unpack(str(acq_settings.frame_length)+"f", raw_data)
        output_queue.put(float_data)
    
    mic.stop_stream()
    print("done acquisition!")

def processing_loop(acquisition_settings, 
                    feed_queue, 
                    stop_value, 
                    main_queue=None):
    
    mean_buff = np.array([0. for i in range(int(1.*acquisition_settings.fs/acquisition_settings.frame_length))])
    df = float(acquisition_settings.fs)/acquisition_settings.frame_length
    
    i = 0
    while not stop_value.value:
        values = feed_queue.get()
        values = np.array(values)
        abs_fft = np.abs(fft(values))
        mean = mean_func(abs_fft, df)
        mean_buff[i%mean_buff.shape[0]] = mean
        
        variance = np.mean(np.power(mean_buff-mean, 2))
        mean_buff_mean = np.mean(mean_buff)
        
        diff = mean - ((1.514-0.000015*variance)*mean_buff_mean)
        
        if i % 6 == 0:
            main_queue.put(diff)
            
        i += 1

        
#simple mean func over abs(fft)
def mean_func(x, df):
    start_index = math.floor(0./df)
    end_index = math.floor(100./df)
    
    mean = np.mean(x[start_index:end_index])
    return mean


if __name__ == "__main__":
    acq_settings = AcquisitionSettings()
    main_queue = Queue()
    q = Queue()
    stop_value = Value('b', False)
    
    acq_process = Process(target=acquisition_loop, args=(acq_settings, q, stop_value))
    proc_process = Process(target=processing_loop, args=(acq_settings, 
                                                         q, 
                                                         stop_value,
                                                         main_queue))
    
    acq_process.start()
    proc_process.start()
    
    for i in range(100):
        val = main_queue.get()
        if val > 0.:
            print("############")
        else:
            print("#")
    
    stop_value.value = True

    