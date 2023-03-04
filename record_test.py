import sounddevice as sd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
device_list = sd.query_devices()
print(device_list)

sd.default.device = [1, 6] # Input, Outputデバイス指定



input_stream = sd.Stream(
        channels=1,
        dtype='float32'
    )
    
output_stream = sd.OutputStream(
        channels=1,
        dtype='float32'
    )

input_stream.start()
output_stream.start()

while True:
    try: 
        start_time = time.time()
        input_data = input_stream.read(2048)
        wav = input_data[0].T
        wav_rmse = np.sqrt(np.square(wav).mean())
        input_stream.write(wav.T)
        
        print("sample rate : ",end="")
        print(input_stream.samplerate)
        print("sample size : ",end="")
        print(input_stream.samplesize)
        print("record time : ",end="")
        print(time.time() - start_time)
        print("wav rmse : ",end="")
        print(wav_rmse)
        print()
    
    except KeyboardInterrupt:
        input_stream.stop()
        input_stream.close()
        output_stream.stop()
        outpit_stream.close()