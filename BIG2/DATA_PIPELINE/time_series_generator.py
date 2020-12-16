import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from IPython import display as ipd

dummy_data = np.arange(1,11,1)
dummy_targets = np.arange(10,110,10)

print(dummy_data)
print(dummy_targets)


timeseries_gen = TimeseriesGenerator(dummy_data,dummy_targets,4)

print(len(timeseries_gen))

print(timeseries_gen[0])
print(timeseries_gen[2])

timeseries_gen2=TimeseriesGenerator(dummy_data,dummy_targets,length=3,batch_size=2)
timeseries_iterator=iter(timeseries_gen2)
print(next(timeseries_iterator))
print(next(timeseries_iterator))


timeseries_gen3=TimeseriesGenerator(dummy_data,dummy_targets,length=3,stride=2,batch_size=2,reverse=True)
timeseries_iterator1=iter(timeseries_gen3)
print(next(timeseries_iterator1))
print(next(timeseries_iterator1))

from scipy.io.wavfile import read,write

rate,song = read("data/mixture.wav")
print(f"rate : {rate}")

print(song.shape)

timeseries_gen4=TimeseriesGenerator(song,targets=song,length=200000,stride=100000,batch_size=1)
iterator= iter(timeseries_gen4)

#print(next(timeseries_iterator))



for i in range(3):
    sample,target=next(iterator)
    print(sample.shape)
    write(f"data/example{i}.wav",rate,sample[0])
    print(f"Sample {i}")
    ipd.display(ipd.Audio(f"data/example{i}.wav"))


timeseries_gen = TimeseriesGenerator(song, song, length=200000, stride=200000, batch_size=1, sampling_rate=2,start_index=200000,shuffle=True,reverse=True)
timeseries_iterator = iter(timeseries_gen)
sample,target=next(timeseries_iterator)
print(sample.shape,target.shape)
