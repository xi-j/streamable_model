import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil as ceil
from copy import deepcopy

import time
import numpy as np
from matplotlib import pyplot as plt


from streamable_model_v3 import StreamableModel


#a simple static convolutional network
class StaticModel(nn.Module):
    def __init__(self, num_layers, config = 0):
        super().__init__()
        self.layers = []

        if config == 0:
            for i in range(num_layers):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 3//2))

            self.layers = nn.Sequential(*self.layers)

        elif config == 1:
            for i in range(num_layers):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 5, stride = 1, padding = 5//2))

            self.layers = nn.Sequential(*self.layers)

        elif config == 2:
            for i in range(num_layers // 2):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 5, stride = 1, padding = 5//2))
            for i in range(num_layers - num_layers // 2):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 3//2))

            self.layers = nn.Sequential(*self.layers)

        # encoder-decoder 1
        elif config == 3:
            self.layers.append(nn.Conv1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2))
            for i in range(num_layers):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 3//2))
            self.layers.append(nn.ConvTranspose1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2, output_padding = 10 - 1))

            self.layers = nn.Sequential(*self.layers)

        # encoder-decoder 2
        elif config == 4:
            self.layers.append(nn.Conv1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2))
            for i in range(num_layers):
                self.layers.append(nn.Conv1d(256, 256, kernel_size = 5, stride = 1, padding = 5//2))
            self.layers.append(nn.ConvTranspose1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2, output_padding = 10 - 1))

            self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

# execute a streamable model with buffer_num buffers under specified configuration and return the average execution time of streamable forward 
def exec_time(num_layers, config = 0, delta_t = 20, fs = 8000, buffer_num = 100, get_params_num = False):
    buffer_length = fs * delta_t // 1000
    sample_num = buffer_length * buffer_num
    my_static_model = StaticModel(num_layers = num_layers, config = config)
    my_streamable_model = StreamableModel(buffer_length, my_static_model.layers)
    test_signal = torch.rand(1, 256, sample_num) 

    # get number of parameters
    if get_params_num:
        numparams = 0
        for i in range(my_streamable_model.num_layers):
            for f in my_streamable_model.streamable_layers[i].parameters():
                if f.requires_grad:
                    numparams += f.numel()
        if config == 0:
            print('{} layers streamable model with kernel_size = 3 trainable Parameters: {}'.format(num_layers, numparams))
        elif config == 1:
            print('{} layers streamable model with kernel_size = 5 trainable Parameters: {}'.format(num_layers, numparams))
        elif config == 2:
            print('{} layers streamable model with mixed kernel size trainable Parameters: {}'.format(num_layers, numparams))


    ts = []
    for i in range(buffer_num):
        input_buffer = test_signal[:, :, buffer_length*i : buffer_length*(i+1)]
        t0 = time.time()
        with torch.no_grad():
            output_buffer, bad_output_length = my_streamable_model(input_buffer)
        t1 = time.time()
        t = (t1 - t0)*1000
        ts.append(t)

    return sum(ts)/len(ts) 



# testing or benchmark
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="test or benchmark streamable model.")
    parser.add_argument("--num_layers","-nl", help = "number of layers", type = int, default = 16)
    parser.add_argument("--frequency","-fs", help = "sampling frequency", type = int, default = 8000)
    parser.add_argument("--delta_t","-dt", help = "buffer corresponding time in ms", type = int, default = 20)
    parser.add_argument("--buffer_num","-bn", help = "total number of buffers", type = int, default = 100)
    parser.add_argument("--mode", "-m", help = "test or benchmark streamable model", type = str, choices = ["test", "plot", "benchmark"], default = "plot")
    parser.add_argument("--config", "-c", help = "test network configuration", type = int, choices = [0,1,2,3,4], default = 3)
    args = parser.parse_args()

    mode = args.mode
    fs = args.frequency
    num_layers = args.num_layers
    delta_t = args.delta_t
    config = args.config
    buffer_length = fs * delta_t // 1000
    buffer_num = args.buffer_num
    sample_num = buffer_length * buffer_num

    if mode == "benchmark":
        print("##############################################################")
        print("sampling frequency: {}Hz".format(fs))
        print("buffer response time: {}ms".format(delta_t))
        print("total number of buffers: {}".format(buffer_num))
        print("number of samples per buffer: {}".format(buffer_length))
        print("total number of samples: {}".format(sample_num))
        print("number of layers: {}".format(num_layers))
        print("##############################################################")

        print("benchmark streamable model...")

        my_static_model = StaticModel(num_layers =  num_layers, config = config)
        my_streamable_model = StreamableModel(buffer_length, my_static_model.layers)
        test_signal = torch.rand(1, 256, sample_num) 
        ts = []
        for i in range(buffer_num):
            input_buffer = test_signal[:, :, buffer_length*i : buffer_length*(i+1)]

            t0 = time.time()
            with torch.no_grad():
                output_buffer, bad_output_length = my_streamable_model(input_buffer)
            t1 = time.time()
            t = (t1 - t0)*1000
            print("execution time of one streamable forward call: {}ms".format(t))
            ts.append(t)

            output_length = output_buffer.shape[2]
            print("No.{} buffer output:".format(i))
            print(output_buffer)
            print("number of bad output samples: ", bad_output_length)
            print("total output length including corrected samples and bad samples:" , output_length)
            print("___________________________________________________________________")

        print("execution time of one streamable forward call: {}ms".format(sum(ts)/len(ts)))

    elif mode == "plot":
        print("##############################################################")
        print("sampling frequency: {}Hz".format(fs))
        print("total number of buffers: {}".format(buffer_num))
        print("number of layers: {}".format(num_layers))
        print("##############################################################")
        t0s = []
        t1s = []
        t2s = []
        t3s = []
        t4s = []
        dts = [20, 25, 30, 35, 40]

        plt.figure()
        for dt in dts:
            t0 = exec_time(num_layers = num_layers, config = 0, delta_t = dt, fs = fs, buffer_num = buffer_num, get_params_num = False)
            t1 = exec_time(num_layers = num_layers, config = 1, delta_t = dt, fs = fs, buffer_num = buffer_num, get_params_num = False)
            t2 = exec_time(num_layers = num_layers, config = 2, delta_t = dt, fs = fs, buffer_num = buffer_num, get_params_num = False)
            t3 = exec_time(num_layers = num_layers, config = 3, delta_t = dt, fs = fs, buffer_num = buffer_num, get_params_num = False)
            t4 = exec_time(num_layers = num_layers, config = 4, delta_t = dt, fs = fs, buffer_num = buffer_num, get_params_num = False)
            t0s.append(t0)
            t1s.append(t1)
            t2s.append(t2)
            t3s.append(t3)
            t4s.append(t4)
            plt.text(dt, t0, str(round(t0,2)))
            plt.text(dt, t1, str(round(t1,2)))
            plt.text(dt, t2, str(round(t2,2)))
            plt.text(dt, t3, str(round(t3,2)))
            plt.text(dt, t4, str(round(t4,2)))

        p0, = plt.plot(dts, t0s, 'o', color='black')
        p1, = plt.plot(dts, t1s, 'o', color='red')
        p2, = plt.plot(dts, t2s, 'o', color='blue')
        p3, = plt.plot(dts, t3s, 'o', color='green')
        p4, = plt.plot(dts, t4s, 'o', color='orange')
        #base, = plt.plot(dts, dts, color='gray')

        plt.title("execution time of streamable forward with {}layers ".format(num_layers))
        plt.xlabel("buffer response time(ms)")
        plt.ylabel("execution time(ms)")
        plt.legend([p0, p1, p2, p3, p4], ['kernel_size = 3', 'kernel_size = 5', 'mixed', 'encoder-decoder(10x) & kernel_size = 3', 'encoder-decoder(10x) & kernel_size = 5'])
        plt.savefig("benchmark of {}-layer network at {}kHz.png".format(num_layers, fs//1000))

        
    elif mode == "test":
        print("##############################################################")
        print("sampling frequency: {}Hz".format(fs))
        print("buffer response time: {}ms".format(delta_t))
        print("total number of buffers: {}".format(buffer_num))
        print("number of samples per buffer: {}".format(buffer_length))
        print("total number of samples: {}".format(sample_num))
        print("number of layers: {}".format(num_layers))
        print("##############################################################")
        print("test streamable model...")

        my_static_model = StaticModel(num_layers = 32, config = config)
        my_streamable_model = StreamableModel(buffer_length, my_static_model.layers)
        test_signal = torch.rand(1, 256, sample_num) 
        output_ref = my_static_model(test_signal)
        output_full = torch.zeros(output_ref.shape)
        output_counter = 0

        for i in range(buffer_num):
            input_buffer = test_signal[:, :, buffer_length*i : buffer_length*(i+1)]
            output_buffer, bad_output_length = my_streamable_model(input_buffer)
            if output_buffer == None:
                continue
            output_length = output_buffer.shape[2]

            # re-run static model, maybe slow
            output_buffer_ref = my_static_model(test_signal[:, :, :buffer_length*(i+1)])[:, :, -output_length:]
            equal = (abs(output_buffer_ref - output_buffer) < 1e-5).all().item()

            print("No.{} buffer output:".format(i))
            print(output_buffer[:, 0, :])
            print("number of bad output samples: ", bad_output_length)
            print("total number of output samples including corrected samples and bad samples:" , output_length)

            if equal:
                print("No.{} buffer output passed".format(i))
            else:
                print("WRONG OUTPUT!!!")
                print("WRONG OUTPUT!!!")
                print("WRONG OUTPUT!!!")
                print("correct output:" ,output_buffer_ref[:, 0, :])

            print("___________________________________________________________________")
            
            output_full[:, :, output_counter:output_counter + output_length] = output_buffer
            output_counter += (output_length - bad_output_length)


        print("reference full output:")
        print(output_ref)

        equal = (abs(output_ref - output_full) < 1e-5).all().item()
        if equal:
            print("full output passed")


