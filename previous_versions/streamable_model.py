import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil as ceil

import time

#StreamableModel
class StreamableModel(nn.Module):
    '''
    wrapper class for conv1d to streamable
    '''
    class StreamableConv1D(nn.Module):
        '''
        @params:
        layer: a conv1d layer
        input_length: length of input signals
        layer_id: index of the layer
        ''' 
        def __init__(self, layer, layer_id, input_length:int):
            super().__init__()

            self.layer = layer
            self.input_length = input_length
            self.in_channels = layer.in_channels
            self.out_channels = layer.out_channels
            self.kernel_size = layer.kernel_size[0]
            self.stride = layer.stride[0]
            self.layer_id = layer_id

            self.buffer = torch.zeros(1, self.in_channels, self.input_length + self.kernel_size - 1)

            self.input_counter = 0
            self.output_counter = 0

        '''
        @params:
        input_tile: input tile, len(tile) == self.input_length
        @return: output tile 
        ''' 
        def forward(self, input_tile):
            if input_tile == None:
                return
            # prev_length := number of useful samples already stored at the end of the buffer
            # prev_length in range [kernel_size - stride, kernel_size - 1] except for first forward prev_length = 0
            prev_length = self.input_counter - self.output_counter * self.stride
            
            input_length = input_tile.shape[2]
            full_length = prev_length + input_length

            # push new samples to the end of the buffer and left shift the rest
            if prev_length > 0:
                self.buffer = torch.roll(self.buffer, -input_length, dims = 2)
            self.buffer[:, :, -input_length:] = input_tile

            # number of samples in the buffer is not big enough for convolution
            if full_length < self.kernel_size:
                self.input_counter += input_length
                return None

            # tiled convolution
            else: 
                output_tile = self.layer(self.buffer[:, :, -full_length:])
                output_length = output_tile.shape[2]
                self.input_counter += input_length
                self.output_counter += output_length 
                return output_tile
 

    '''
    @params:
    tile_length: number of samples per forward call to the first layer
    layers: a sequence of conv1d layers
    '''
    def __init__(self, tile_length:int, layers:torch.nn.Sequential):  
        super().__init__()

        self.num_layers = len(layers)
        self.streamable_layers = []

        input_length = tile_length
        self.input_length = input_length

        # streamalize all conv1d layers
        for i in range(self.num_layers):
            kernel_size = layers[i].kernel_size[0]
            stride = layers[i].stride[0]
            padding = layers[i].padding[0]
            dilation = layers[i].dilation[0]

            self.streamable_layers.append(self.StreamableConv1D(layers[i], i, input_length))
            output_length = ceil((input_length + kernel_size - 1 + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1)
            input_length =  output_length

            self.in_channels = self.streamable_layers[0].in_channels
            self.out_channels = self.streamable_layers[-1].out_channels

    def __len__(self):
        return self.num_layers

    def __getitem__(self, i):
        return self.streamable_layers[i]

    '''
    @params:
    tile: input tile, len(tile) == self.input_length
    @return: output tile
    ''' 
    def forward(self, tile):
        for i in range(self.num_layers):
            tile = self.streamable_layers[i](tile)

        return tile


# testing
if __name__ == '__main__':
    tile_length = 20
    tile_num = 100
    sample_num  = tile_length * tile_num


    my_model = nn.Sequential(nn.Conv1d(3, 3, kernel_size = 7, stride = 2), nn.Conv1d(3, 3, kernel_size = 3, stride = 4), nn.Conv1d(3, 3, kernel_size = 1, stride = 2), nn.Conv1d(3, 3, kernel_size = 3, stride = 1), nn.Conv1d(3, 1, kernel_size = 5, stride = 4))

    my_streamable_model = StreamableModel(tile_length, my_model)

    test_signal = torch.randn(1, 3, sample_num) 
    output_ref = my_model(test_signal)
    output_counter = 0

    print("##############################################################")
    print("input", tile_num, "tiles of", tile_length, "samples each:")
    print("##############################################################")
    for i in range(tile_num):
        output_tile = my_streamable_model(test_signal[:, :, tile_length*i : tile_length*(i+1)])

        if output_tile == None:
            continue

        output_tile_length = output_tile.shape[2]
        print("{}th tile output:".format(i))
        print(output_tile)
        equal = (abs(output_ref[:, :, output_counter:output_counter+output_tile_length] - output_tile) < 1e-4).all().item()
        if equal:
            print("{}th tile output passed".format(i))
        else:
            print("WRONG OUTPUT!!!")
            print("WRONG OUTPUT!!!")
            print("WRONG OUTPUT!!!")
        output_counter += output_tile_length

        time.sleep(0.01)

    print("reference full output:")
    print(output_ref)


