import sounddevice as sd
import numpy as np

device_list = sd.query_devices()
#print(device_list)

print(sd.default.device)

for device_number in sd.default.device:
    print(device_number)
    #print(device_list[device_number])

duration=3

def callback(indata, outdata, frames, time, status):
    print("frames : ",end="")
    print(frames)
    print("time : ",end="")
    print(time)
    print(status)
    n_samples, n_channels = outdata.shape
    print("n_sample shape : ",end="")
    print(n_samples)
    print("n channels : ",end="")
    print(n_channels)
    print(indata.shape)
    outdata[:] = indata

    
stream = sd.Stream(
        samplerate=24000,
        channels=1,
        blocksize=2048,     # The number of frames passed to the stream callback function.
        dtype='float32', 
        callback=callback
    )

stream.start()
sd.sleep(int(duration * 1000))
stream.stop()
sd.sleep(int(1000))
stream.start()
sd.sleep(int(duration * 1000))
stream.stop()

"""   
class sound_stream:
    def __init__(self, sampling_rate, channels):
        self.sampling_rate = sampling_rate
        self.channels = channels
        
        self.stream_flag = 0
        
    def stream_start(self):
    
    def callback(self, indata, outdata, frames, time, status):
        outdata[:] = indata
"""