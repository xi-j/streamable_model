import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil as ceil
from copy import deepcopy

import time

#streamable convolutional network
class StreamableModel(nn.Module):
    '''
    wrapper class for streamable conv1d 
    '''
    class StreamableConv1D(nn.Module):
        '''
        @params:
        layer: a conv1d layer
        input_length: length of input signals
        layer_id: index of the layer
        ''' 
        def __init__(self, layer:nn.Conv1d, input_length:int, layer_id:int):
            super().__init__()

            self.kernel = deepcopy(layer)
            self.kernel.padding = 0

            self.input_length = input_length
            self.in_channels = layer.in_channels
            self.out_channels = layer.out_channels
            self.kernel_size = layer.kernel_size[0]
            self.stride = layer.stride[0]
            self.padding = self.kernel_size // 2

            # number of samples in self.buffer dependent on padding zeros from previous layers
            self.bad_input_length = 0

            self.layer_id = layer_id

            self.buffer = torch.zeros(1, self.in_channels, self.input_length + self.kernel_size - 1 + self.padding)

            # number of good input/output samples so far
            self.input_counter = 0
            self.output_counter = 0


        '''
        @params:
        input_buffer: input buffer, len(buffer) == self.input_length
        bad_input_length: number of samples dependent on padding zeros at the end of input_buffer
        @return: output buffer and number of samples dependent on padding zeros at the end of the output buffer
        ''' 
        def forward(self, input_buffer, bad_input_length = 0):
            if input_buffer == None:
                return
 
            # prev_length := number of useful samples from previous buffers already in the buffer
            prev_length = self.input_counter - self.output_counter * self.stride + self.padding + self.bad_input_length

            # for all input buffers except the first:
            #  |        corrected old samples     |                       new good samples                  |  new bad samples  |
            #  |        self.bad_input_length     | input_length - self.bad_input_length - bad_input_length |  bad_input_length |
            #  | input_length - new_input_length  |                             new_input_length                                |
            
            # for the first input buffer:
            #  |                       new good samples                  |  new bad samples  |
            #  |             input_length - bad_input_length             |  bad_input_length |
            #  |                       input_length = new_input_length                       |

            input_length = input_buffer.shape[2] 
            new_input_length = input_length - self.bad_input_length

            full_length = prev_length + new_input_length + self.padding
            good_length = full_length - bad_input_length - self.padding

            # buffer initially
            #  |              old good samples           |     old bad samples   | paddings |
            #                                            | self.bad_input_length |

            # left shift samples already in the buffer
            #  |   ...   | old bad samples |      reserved for input_buffer      | paddings |
            if self.padding:
                self.buffer[:, :, :-self.padding] = torch.roll(self.buffer[:, :, :-self.padding], -new_input_length , dims = 2)
            else:
                self.buffer = torch.roll(self.buffer, -new_input_length , dims = 2)


            # push new samples to the end of the buffer
            #  |   ...   |  old correct samples  |                       new good samples                  |  new bad samples  | paddings |
            #  |   ...   | self.bad_input_length |            new_input_length - bad_input_length          |  bad_input_length | paddings |
            #       |                               full_length = prev_length + new_input_length + padding                                |        
            if self.padding: 
                self.buffer[:, :,- (input_length + self.padding) : - self.padding] = input_buffer
            else:
                self.buffer[:, :,-input_length : ] = input_buffer

            # number of samples in the buffer is not big enough for convolution
            if full_length < self.kernel_size:
                self.input_counter += (input_length - bad_input_length)
                return None

            # buffer convolution
            else: 
                output_buffer = self.kernel(self.buffer[:, :, -full_length:])
                output_length = output_buffer.shape[2]
                good_output_length = max(0, (int)((good_length - self.kernel_size)/self.stride + 1))

                bad_output_length = output_length - good_output_length

                self.input_counter += (input_length - bad_input_length)
                self.output_counter += (output_length - bad_output_length)
                self.bad_input_length = bad_input_length

                return output_buffer, bad_output_length


    '''
    @params:
    buffer_length: number of samples per forward call to the first layer
    layers: a sequence of conv1d layers
    '''
    def __init__(self, buffer_length:int, layers:torch.nn.Sequential):  
        super().__init__()

        self.num_layers = len(layers)
        self.streamable_layers = []

        input_length = buffer_length
        self.input_length = input_length

        # streamalize all conv1d layers
        for i in range(self.num_layers):
            kernel_size = layers[i].kernel_size[0]
            stride = layers[i].stride[0]
            padding = kernel_size // 2

            self.streamable_layers.append(self.StreamableConv1D(layers[i], input_length, i))
            output_length = ceil((input_length + kernel_size - 1  - 1)/stride + 1)
            input_length = output_length

        self.in_channels = self.streamable_layers[0].in_channels
        self.out_channels = self.streamable_layers[-1].out_channels

    def __len__(self):
        return self.num_layers

    def __getitem__(self, i):
        return self.streamable_layers[i]


    '''
    @params:
    buffer: input buffer, len(buffer) == self.input_length
    @return: output buffer and number of samples dependent on padding zeros at the end of the output buffer
    ''' 
    def forward(self, buffer):
        bad_input_length = 0
        for i in range(self.num_layers):
            buffer, bad_output_length = self.streamable_layers[i](buffer, bad_input_length)
            bad_input_length = bad_output_length

        return buffer, bad_output_length


# testing
if __name__ == '__main__':
    buffer_length = 10
    buffer_num = 20
    sample_num  = buffer_length * buffer_num

    
    #my_model = nn.Sequential(nn.Conv1d(3, 1, kernel_size = 5, stride = 1, padding = 5//2))
    #my_model = nn.Sequential(nn.Conv1d(3, 4, kernel_size = 7, stride = 1, padding = 7//2),nn.Conv1d(4, 4, kernel_size = 5, stride = 1, padding = 5//2), nn.Conv1d(4, 4, kernel_size = 1, stride = 1, padding = 1//2),nn.Conv1d(4, 1, kernel_size = 3, stride = 1, padding = 3//2))
    my_model = nn.Sequential(nn.Conv1d(3, 4, kernel_size = 7, stride = 1, padding = 7//2), nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), nn.Conv1d(4, 4, kernel_size = 5, stride = 1, padding = 5//2), nn.Conv1d(4, 4, kernel_size = 1, stride = 1, padding = 1//2), nn.Conv1d(4, 1, kernel_size = 3, stride = 1, padding = 3//2))
    #my_model = nn.Sequential(nn.Conv1d(3, 4, kernel_size = 7, stride = 2, padding = 7//2), nn.Conv1d(4, 4, kernel_size = 1, stride = 1, padding = 1//2), nn.Conv1d(4, 4, kernel_size = 3, stride = 2, padding = 3//2), nn.Conv1d(4, 1, kernel_size = 5, stride = 3, padding = 5//2))

    my_streamable_model = StreamableModel(buffer_length, my_model)

    test_signal = torch.rand(1, 3, sample_num) 
    output_ref = my_model(test_signal)
    output_full = torch.zeros(output_ref.shape)
    output_counter = 0


    print("##############################################################")
    print("buffer_num:", buffer_num, "buffer_length:", buffer_length)
    print("##############################################################")
    for i in range(buffer_num):
        input_buffer = test_signal[:, :, buffer_length*i : buffer_length*(i+1)]
        output_buffer, bad_output_length = my_streamable_model(input_buffer)
        if output_buffer == None:
            continue
        output_length = output_buffer.shape[2]

        # re-run static model up to buffer_length*(i+1) samples 
        output_buffer_ref = my_model(test_signal[:, :, :buffer_length*(i+1)])[:, :, -output_length:]
        equal = (abs(output_buffer_ref - output_buffer) < 1e-5).all().item()

        print("No.{} buffer output:".format(i))
        print(output_buffer)
        print("number of bad output samples: ", bad_output_length)
        print("total output length including corrected samples and bad samples:" , output_length)

        if equal:
            print("No.{} buffer output passed".format(i))
        else:
            print("WRONG OUTPUT!!!")
            print("WRONG OUTPUT!!!")
            print("WRONG OUTPUT!!!")
            print("correct output:" ,output_buffer_ref)

        print("___________________________________________________________________")
         
        output_full[:, :, output_counter:output_counter + output_length] = output_buffer
        output_counter += (output_length - bad_output_length)
        time.sleep(0.01)

    print("reference full output:")
    print(output_ref)

    equal = (abs(output_ref - output_full) < 1e-5).all().item()
    if equal:
        print("full output passed")

    '''
    equal = abs(output_ref[:, :, output_counter:output_counter + output_length] - output_buffer) < 1e-5
    print(equal)
    incorrect = torch.sum(~equal).item()
    check = (incorrect <=  my_streamable_model.out_channels * bad_output_length) 
    '''

