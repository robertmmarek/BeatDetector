import pyaudio
import struct

from matplotlib import pyplot as pp

p = pyaudio.PyAudio()

microphone = p.open(format=pyaudio.paFloat32,
                    rate=10000,
                    channels=1,
                    input=True,
                    )

frame_size = 2000
nb_of_frames = 20

microphone.start_stream()
test_read = microphone.read(frame_size*nb_of_frames)
decoded = struct.unpack(str(frame_size*nb_of_frames)+"f", test_read)

microphone.stop_stream()
pp.figure()
pp.plot(decoded)