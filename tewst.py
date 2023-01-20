import sounddevice as sd
import torch

# outputData = torch.zeros([1,8000])

def callback(indata, frames, time, status):
    indata = torch.tensor(indata)
    print(indata.shape)
    indata = indata.T
    assert indata.shape == outputData.shape, print(indata.shape, outputData.shape)
    outputData = indata

    # print(np.array(indata))

stream = sd.InputStream(callback=callback, blocksize=8000, samplerate = 8000, channels=1)
stream.start()

import time 

time.sleep(1000)

stream.stop()