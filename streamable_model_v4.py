import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil as ceil
from copy import deepcopy

import time

'''
streamable convolutional network
NOTE: 
1. support centered convolution, centered transpose convolution and constant interpolation 
   centered mains odd kernel_size, padding = kernel_size // 2 and output_padding = stride - 1 for transpose convolution 
   support stride, groups; not support dilation
2. forward function accepts a fixed-length buffer of samples and returns a buffer of samples and an integer bad_output_length.
   bad samples are samples dependent on padding, not yet fully computed or dependent on other bad samples
   fully-computed, never-changed samples are good samples; corrected samples are good samples used to replace bad samples.
   returned buffer has 3 parts: corrected samples, good samples and bad samples (except no corrected samples for the first forward)
   bad_output_length is the number of bad samples at the end of the buffer
   (i+1)th returned buffer's corrected samples are the fully computed ith returned buffer's bad samples
   so, the caller should replace previously returned bad samples with new corrected samples for the most accurate result
'''
class StreamableModel(nn.Module):
    '''
    streamable nn.Conv1d
    NOTE: padding = odd kernel_size and kernel_size // 2 are enforced. 
    s.t. output_length = (number of corrected samples) + ceil(input_length/stride)
    dilation and groups are not supported
    '''
    class StreamableConv1D(nn.Module):
        '''
        @params:
        layer: a nn.Conv1d layer with odd kernel_size and padding = kernel_size // 2 
        input_length: length of input signals
        layer_id: index of the layer
        ''' 
        def __init__(self, layer:nn.Conv1d, input_length:int, layer_id:int):
            super().__init__()
            self.input_length = input_length
            self.layer_id = layer_id

            # a kernel with zero padding, padding is handled manually
            self.kernel = deepcopy(layer)
            self.kernel.padding = 0

            self.in_channels = layer.in_channels
            self.out_channels = layer.out_channels
            self.kernel_size = layer.kernel_size[0]
            self.stride = layer.stride[0]
            self.padding = self.kernel_size // 2
            self.groups = layer.groups


            # number of bad input samples from the previous layer in the last forward call
            self.bad_input_length = 0
            # number of bad output samples produced in the last forward call
            self.bad_output_length = 0

            # store previous samples and new samples
            self.buffer_full_length = self.input_length + self.kernel_size - 1 + self.padding
            self.buffer = torch.zeros(1, self.in_channels, self.buffer_full_length)

            # number of good input/output samples so far
            self.input_counter = 0
            self.output_counter = 0
    

        '''
        @params:
        input_buffer: input buffer with self.input_length samples
        bad_input_length: number of samples not fully computed at the end of input buffer
        @return: output buffer and number of samples not fully computed at the end of the output buffer
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

                self.bad_output_length = bad_output_length
                return output_buffer, bad_output_length

    '''
    streamable constant interpolation
    '''
    class StreambleConstInterpolation(nn.Module):
        '''
        @params:
        layer: a nn.Upsample layer
        input_length: length of input signals
        layer_id: index of the layer

        NOTE: only support 'nearest' mode now. caller should initialize layer with integer scale_factor
        ''' 
        def __init__(self, layer:nn.Upsample, input_length:int, layer_id:int):
            super().__init__()
            self.input_length = input_length
            self.layer_id = layer_id

            self.interp = deepcopy(layer)
            self.scale_factor = (int) (layer.scale_factor)
            self.mode = layer.mode
            self.output_length = input_length * self.scale_factor

        '''
        @params:
        input_buffer: input buffer with self.input_length samples
        bad_input_length: number of samples not fully computed at the end of input buffer
        @return: output buffer and number of samples not fully computed at the end of the output buffer
        ''' 
        def forward(self, input_buffer, bad_input_length = 0):
            output_buffer = self.interp(input_buffer)
            bad_output_length = bad_input_length * self.scale_factor

            return output_buffer, bad_output_length

    '''
    streamable activation e.g. ReLU, PReLU
    '''
    class StreamableActivation(nn.Module):
        def __init__(self, layer:nn.ReLU, input_length:int, layer_id:int):
            super().__init__()
            self.input_length = input_length
            self.layer_id = layer_id

            self.activation = deepcopy(layer)

        '''
        @params:
        input_buffer: input buffer with self.input_length samples
        bad_input_length: number of samples not fully computed at the end of input buffer
        @return: output buffer and number of samples not fully computed at the end of the output buffer
        ''' 
        def forward(self, input_buffer, bad_input_length = 0):
            output_buffer = self.activation(input_buffer)
            bad_output_length = bad_input_length

            return output_buffer, bad_output_length


    '''
    streamable nn.ConvTranspose1d
    NOTE: odd kernel_size, padding = kernel_size // 2, output_padding = stride - 1 and stride < kernel_size are enforced. 
    s.t. output_length = (number of corrected samples) + input_length * stride 
    dilation and groups are not supported
    '''
    class StreamableConvT1D(nn.Module):
        '''
        @params:
        layer: a nn.ConvTranspose1d layer with odd kernel_size, padding = kernel_size // 2 , output_padding = stride - 1 and stride < kernel_size 
        input_length: length of input signals
        layer_id: index of the layer
        ''' 
        def __init__(self, layer:nn.ConvTranspose1d, input_length:int, layer_id:int):
            super().__init__()

            self.input_length = input_length
            self.layer_id = layer_id

            self.in_channels = layer.in_channels
            self.out_channels = layer.out_channels

            # create a kernel with zero padding, output_padding and bias
            # padding, output_padding and bias are handled manually
            self.kernel = deepcopy(layer)
            self.kernel.padding = 0
            self.kernel.output_padding = 0

            # conv transpose is computed without bias. bias is added to the output when returned to avoid duplicated bias addition
            if self.kernel.bias!= None:
                self.bias = self.kernel.bias.detach().clone().view(1, self.out_channels, 1)
            else:
                self.bias = 0
            self.kernel.bias = nn.Parameter(torch.zeros(self.out_channels))

            self.kernel_size = layer.kernel_size[0]
            self.stride = layer.stride[0]
            self.padding = self.kernel_size // 2
            self.output_padding = self.stride - 1

            # self.overlap_length := number of output samples dependent on both two adjacent samples
            self.overlap_length = max(0, self.kernel_size - self.stride)

            '''
            bookkeeping variables
            '''
            # number of input samples not fully computed received from the previous layer in the last forward call
            self.bad_input_length = 0

            # index of the first bad/corrected output samples in self.buffer0 and self.buffer
            self.bad_output_start = 0

            # number of bad output samples returned to the caller in the last forward call
            self.bad_output_length = 0

            # store the transpose conv output of bad input samples from the last forward call
            self.last_bad_output = None
            
            # True for the first time to call forward(), False otherwise
            self.first_forward = True

            # number of good output samples produced so far
            self.good_output_counter = -self.padding

            '''
            output buffer related variables
            '''
            # self.valid_length :=  the total # of output samples in self.buffer dependent on the input 
            self.valid_length = self.kernel_size + (self.input_length  - 1) * self.stride

            # self.right_discard_length := # of output samples at the end of the buffer to throw away if self.padding > self.output_padding
            #                        or # of padding(=bias) to append at the end if the output signal if self.padding < self.output_padding
            self.right_discard_length = self.padding - self.output_padding


            self.buffer_full_length = self.valid_length + max(0, self.right_discard_length)
            
            # the index+1 of the last sample in the buffer
            self.buffer_end = 0

            # buffer to store output samples
            self.buffer = torch.zeros(1, self.out_channels, self.buffer_full_length)



        '''
        @params:
        input_buffer: input buffer with self.input_length samples
        bad_input_length: number of bad input samples at the end of input_buffer
        @return: output buffer and number of samples not fully computed at the end of the output buffer
        ''' 
        def forward(self, input_buffer, bad_input_length = 0):
            if input_buffer == None:
                return

            input_length = input_buffer.shape[2]

            # 1. for all input buffers except the first with large enough input_length s.t. good_input >= 0 
            # |  corrected_input = corrected_good_input    |           good_input        |     bad_input    |
            # 2. for all input buffers except the first with very small input_length 
            #     s.t. good_input = 0, corrected_bad_input > 0, which implies not all corrected samples are good
            # | corrected_good_input | corrected_bad_input |                   new_bad_input                |
            # |                corrected_input             |  
            #                        |                                bad_input                             |
            # 3. for the first input buffer
            # |                  good_input                |                    bad_input                   |

            corrected_input_length = self.bad_input_length
            self.bad_input_length = bad_input_length

            good_input_length = max(0, input_length - corrected_input_length - bad_input_length)
            new_bad_input_length = input_length - corrected_input_length - good_input_length
            corrected_bad_input_length = bad_input_length - new_bad_input_length
            corrected_good_input_length = corrected_input_length - corrected_bad_input_length


            # compute corrected_good_output, good_output and bad_output, 
            # i.e. transpose conv output of corrected_good_input, good_input and bad_input respectively

            if corrected_good_input_length > 0:
                corrected_good_input = input_buffer[:, :, 0 : corrected_good_input_length]
                corrected_good_output = self.kernel(corrected_good_input)
                corrected_good_output_length = corrected_good_output.shape[2]
            else:
                corrected_good_input = None
                corrected_good_output = None
                corrected_good_output_length = 0

            if corrected_input_length > 0:
                corrected_output_length = (corrected_input_length - 1) * self.stride + self.kernel_size
            else:
                corrected_output_length = 0

            if good_input_length > 0:
                good_input = input_buffer[:, :, corrected_input_length : corrected_input_length + good_input_length]
                good_output = self.kernel(good_input)
                good_output_length = good_output.shape[2]
            else:
                good_input = None
                good_output = None
                good_output_length = 0

            if bad_input_length > 0:
                bad_input = input_buffer[:, :, -bad_input_length:]
                bad_output = self.kernel(bad_input)
                bad_output_length = bad_output.shape[2]
            else:
                bad_input = None
                bad_output = None
                bad_output_length = 0


            if self.first_forward:
                self.first_forward = False

                # 1. add new good output samples to self.buffer
                if good_output_length > 0:
                    self.buffer[:, :, 0 : good_output_length] += good_output
                    self.buffer_end = good_output_length

                # 2. add new bad output samples to self.buffer; store new bad output samples as self.last_bad_output
                if bad_output_length > 0:
                    self.bad_output_start = max(0, self.buffer_end - self.overlap_length)
                    self.last_bad_output = bad_output.detach().clone()
                    self.buffer[:, :, self.bad_output_start : self.bad_output_start + bad_output_length] += bad_output
                    self.buffer_end = self.bad_output_start + bad_output_length

                # last step: return part of self.buffer(excluding paddings) to the caller and the number of bad output samples
                output_buffer = self.buffer[:, :, self.padding : self.buffer_end - self.right_discard_length] + self.bias
                output_length = output_buffer.shape[2]

                # self.bad_output_length = | output samples depending on bad input samples U output samples not fully computed U output padding | 
                if bad_output_length > 0:
                    self.bad_output_length = min(output_length, self.buffer_end - self.right_discard_length - self.bad_output_start)

                else: # only when the transpose convolution is the first layer i.e. all input samples are good
                    self.bad_output_length = min(output_length, self.padding)

                self.good_output_counter += self.bad_output_start

                return output_buffer, self.bad_output_length
            
            else:
                # 0. replace (subtract & add) old bad output samples by corrected good output samples, and left shift self.buffer
                if corrected_output_length > 0:
                    self.buffer[:, :, self.bad_output_start : self.bad_output_start + corrected_output_length] -= self.last_bad_output[:, :, 0:corrected_output_length]

                    if corrected_good_output_length > 0:
                        self.buffer[:, :, self.bad_output_start : self.bad_output_start + corrected_good_output_length] += corrected_good_output


                    self.buffer = torch.roll(self.buffer, -self.bad_output_start , dims = 2)
                    self.buffer_end = corrected_good_output_length
                    self.buffer[:, :, corrected_output_length:] = 0

                else: # only when the transpose convolution is the first layer i.e. all input samples are good
                    self.buffer = torch.roll(self.buffer, -(self.buffer_end - self.overlap_length) , dims = 2)
                    self.buffer[:, :, self.overlap_length:] = 0
                    self.buffer_end = self.overlap_length


                # 1. add new good output samples to self.buffer
                if good_output_length > 0:
                    good_output_start = max(0, self.buffer_end - self.overlap_length)
                    self.buffer[:, :, good_output_start : good_output_start + good_output_length] += good_output
                    self.buffer_end = good_output_start + good_output_length

                # 2. add new bad output samples to self.buffer; store new bad output samples as self.last_bad_output
                if bad_output_length > 0:
                    bad_output_start = max(0, self.buffer_end - self.overlap_length)
                    self.buffer[:, :, bad_output_start : bad_output_start + bad_output_length] += bad_output
                    self.buffer_end = bad_output_start + bad_output_length

                    # self.bad_output_start is where to replace bad output samples in the next forward call
                    self.bad_output_start = self.buffer_end - bad_output_length
                    self.last_bad_output = bad_output.detach().clone()


                # last step: return part of self.buffer(excluding paddings) to the caller and the number of bad output samples

                # handle the corner case when the first forward does not even output self.padding good samples 
                if self.good_output_counter < 0: # so in the next forward(s), first some corrected good samples should serve as left padding to throw away
                    output_buffer = self.buffer[:, :, -self.good_output_counter : self.buffer_end - self.right_discard_length] + self.bias

                else:
                    output_buffer = self.buffer[:, :, 0 : self.buffer_end - self.right_discard_length] + self.bias
                    
                output_length = output_buffer.shape[2]

                # self.bad_output_length = | output samples depending on bad input samples U output samples not fully computed U right padding | 
                if bad_output_length > 0:
                    self.bad_output_length = min(output_length, self.buffer_end - self.right_discard_length - self.bad_output_start)
                
                else: # only when the transpose convolution is the first layer i.e. all input samples are good
                    self.bad_output_length = min(output_length, self.padding)

                self.good_output_counter += self.bad_output_start

                return output_buffer, self.bad_output_length




    '''
    @params:
    input_length: number of samples per forward call to the first layer
    layers: a sequence of nn.Conv1d or nn.Upsample layers
    '''
    def __init__(self, input_length:int, layers:nn.Sequential, shortcutDict = None, buffer_length_ratio = 2):  
        super().__init__()

        # a container of streamable layer
        self.streamable_layers = []
        self.num_layers = len(layers)

        self.input_length = input_length
        self.buffer_length_ratio = buffer_length_ratio

        # compute the maximum number of bad input/output samples received/produced by each layer in stable condition
        # compute the number of new input samples received/produced by each layer in stable condition
        # streamalize all layers
        self.bad_samples_num = {}
        self.bad_samples_num[0] = 0
        self.new_samples_num = {}
        self.new_samples_num[0] = self.input_length
        self.total_samples_num = {}
        self.total_samples_num[0] = self.input_length

        new_input_length = self.input_length
        for i in range(0, self.num_layers):
            bad_input_length = self.bad_samples_num[i]
            new_input_length = self.new_samples_num[i]

            # streamalize nn.Conv1d
            if isinstance(layers[i], nn.Conv1d):
                kernel_size = layers[i].kernel_size[0]
                stride = layers[i].stride[0]
                padding = kernel_size // 2

                full_input_length = new_input_length  + padding
                good_input_length = new_input_length  - bad_input_length

                full_output_length = (int)((full_input_length - padding - 1)/stride + 1)
                good_output_length = (int)((good_input_length - padding - 1)/stride + 1)
                bad_output_length = full_output_length - good_output_length
                new_output_length  = ceil(new_input_length / stride)
                total_output_length = bad_output_length + new_output_length

                self.bad_samples_num[i + 1] = bad_output_length
                self.new_samples_num[i + 1] = new_output_length
                self.total_samples_num[i + 1] = total_output_length

                self.streamable_layers.append(self.StreamableConv1D(layers[i], (bad_input_length + new_input_length)*buffer_length_ratio, i))

            # streamalize nn.ConvTranspose1d
            elif isinstance(layers[i], nn.ConvTranspose1d):
                kernel_size = layers[i].kernel_size[0]
                stride = layers[i].stride[0]
                padding = kernel_size // 2

                bad_output_length = bad_input_length * stride + padding
                new_output_length = new_input_length * stride
                total_output_length = bad_output_length + new_output_length

                self.bad_samples_num[i + 1] = bad_output_length
                self.new_samples_num[i + 1] = new_output_length
                self.total_samples_num[i + 1] = total_output_length

                self.streamable_layers.append(self.StreamableConvT1D(layers[i], (bad_input_length + new_input_length)*buffer_length_ratio, i))
                
            # streamalize nn.Upsample with mode = 'nearest'
            elif isinstance(layers[i], nn.Upsample) and layers[i].mode == 'nearest':
                scale_factor = (int) (layers[i].scale_factor)

                bad_output_length = bad_input_length * scale_factor
                new_output_length = new_input_length * scale_factor
                total_output_length = bad_output_length + new_output_length
                
                self.bad_samples_num[i + 1] = bad_output_length
                self.new_samples_num[i + 1] = new_output_length
                self.total_samples_num[i + 1] = total_output_length

                self.streamable_layers.append(self.StreambleConstInterpolation(layers[i], bad_input_length + new_input_length, i))

            # streamalize activations 
            elif isinstance(layers[i], nn.ReLU) or isinstance(layers[i], nn.PReLU):

                bad_output_length = bad_input_length
                new_output_length = new_input_length
                total_output_length = bad_output_length + new_output_length

                self.bad_samples_num[i + 1] = bad_output_length
                self.new_samples_num[i + 1] = new_output_length
                self.total_samples_num[i + 1] = total_output_length

                self.streamable_layers.append(self.StreamableActivation(layers[i], bad_input_length + new_input_length, i))


        # allocate space for residual connections
        self.residual_flag = bool(shortcutDict)
        if self.residual_flag:
            # self.shortcutDict[i] = j means ith layer's input is connected by jth layer's output, i - j > 1
            self.shortcutDict = shortcutDict.copy()

            # save output buffer of residual layer
            self.buffer_img = {}
            self.residual_bad_samples_num = {}
            for i in self.shortcutDict.keys():
                j = self.shortcutDict[i]
                length = self.total_samples_num[i]

                if isinstance(self.streamable_layers[i], StreamableModel.StreamableConv1D) or isinstance(self.streamable_layers[i], StreamableModel.StreamableConvT1D):
                    in_channels = self.streamable_layers[i].in_channels
                elif isinstance(self.streamable_layers[j], StreamableModel.StreamableConv1D) or isinstance(self.streamable_layers[j], StreamableModel.StreamableConvT1D):
                    in_channels = self.streamable_layers[j].out_channels
                elif i - 1 > 0 and (isinstance(self.streamable_layers[i-1], StreamableModel.StreamableConv1D) or isinstance(self.streamable_layers[i-1], StreamableModel.StreamableConvT1D)):
                    in_channels = self.streamable_layers[i - 1].in_channels
                elif j - 1 > 0 and (isinstance(self.streamable_layers[j-1], StreamableModel.StreamableConv1D) or isinstance(self.streamable_layers[j-1], StreamableModel.StreamableConvT1D)):
                    in_channels = self.streamable_layers[j - 1].out_channels

                self.buffer_img[j] = torch.zeros(1, in_channels, length*buffer_length_ratio)
                self.residual_bad_samples_num[j] = 0

    def __len__(self):
        return self.num_layers

    def __getitem__(self, i):
        return self.streamable_layers[i]



    '''
    @params:
    buffer: input buffer with self.input_length samples
    @return: output buffer and number of samples not fully computed at the end of the output buffer
    ''' 
    def forward(self, buffer):
        if self.residual_flag:
            return self.residual_forward(buffer)

        bad_input_length = 0
        for i in range(self.num_layers):
            buffer, bad_output_length = self.streamable_layers[i](buffer, bad_input_length)
            bad_input_length = bad_output_length


        return buffer, bad_output_length


    '''
    @params:
    buffer: input buffer with self.input_length samples
    @return: output buffer and number of samples not fully computed at the end of the output buffer
    ''' 
    def residual_forward(self, buffer):
        bad_input_length = 0
        for i in range(self.num_layers):

            # skip connections: output of layer j connects to input of layer i 
            if i in self.shortcutDict:

                j = self.shortcutDict[i]

                latest_length = buffer.shape[2]
                residual = self.buffer_img[j][:, :, -latest_length:]

                residual_bad_input_length = self.residual_bad_samples_num[j]

                buffer, bad_output_length = self.streamable_layers[i](buffer + residual, max(residual_bad_input_length, bad_input_length))
                bad_input_length = bad_output_length

            else:
                buffer, bad_output_length = self.streamable_layers[i](buffer, bad_input_length)
                bad_input_length = bad_output_length

            # copy residuals 
            if i in self.buffer_img:

                residual_length = buffer.shape[2]
                bad_residual_length =  self.residual_bad_samples_num[i]
                good_residual_length = residual_length - bad_residual_length

                self.buffer_img[i] = torch.roll(self.buffer_img[i], -good_residual_length, dims = 2)
                self.buffer_img[i][:, :, -residual_length:] = buffer.clone()
                self.residual_bad_samples_num[i] = bad_input_length


        return buffer, bad_output_length




# not part of streamable model, used for testing
def residual_forward(layers: nn.Sequential, shortcutDict: dict, x):
    if shortcutDict == None:
        return layers(x)

    buffer_img = dict.fromkeys(shortcutDict.values(), 0)
    num_layers = len(layers)

    for i in range(num_layers):
        if i in shortcutDict:
            j = shortcutDict[i]
            residual = buffer_img[j]
            x = layers[i](x + residual)

        else:
            x = layers[i](x)

        if i in buffer_img:
            buffer_img[i] = x.clone()

    return x


###########################################################################################################################################################################################################################
###########################################################################################################################################################################################################################
# testing
if __name__ == '__main__':

    buffer_length = 24 * 4
    buffer_num = 100
    sample_num  = buffer_length * buffer_num

    '''
    my_model = nn.Sequential(nn.ConvTranspose1d(3, 1, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 2 - 1))

    my_model = nn.Sequential(nn.Conv1d(1, 4, kernel_size = 7, stride = 3, padding = 7//2), nn.Conv1d(4, 4, kernel_size = 5, stride = 2, padding = 5//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 2, padding = 3//2), nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), nn.Conv1d(4, 1, kernel_size = 3, stride = 1, padding = 3//2))


    my_model = []
    my_model.append(nn.Conv1d(1, 256, kernel_size = 21, stride = 10, padding = 21//2))
    for i in range(32):
        my_model.append(nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 3//2, groups = 256))
    my_model.append(nn.ConvTranspose1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2, output_padding = 10 - 1))
    my_model= nn.Sequential(*my_model)

    my_model = nn.Sequential(nn.Conv1d(1, 4, kernel_size = 9, stride = 4, padding = 9//2), \
                             nn.Conv1d(4, 4, kernel_size = 7, stride = 3, padding = 7//2), \
                             nn.Conv1d(4, 4, kernel_size = 5, stride = 2, padding = 5//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.ConvTranspose1d(4, 4, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 2-1), \
                             nn.ConvTranspose1d(4, 4, kernel_size = 7, stride = 3, padding = 7//2, output_padding = 3-1), \
                             nn.ConvTranspose1d(4, 1, kernel_size = 9, stride = 4, padding = 9//2, output_padding = 4-1))
                            
   
    '''
    my_model = []
    my_model.append(nn.Conv1d(1, 256, kernel_size = 21, stride = 10, padding = 21//2))
    my_model.append(nn.ReLU())
    for i in range(10):
        my_model.append(nn.Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 3//2, groups = 256))
        my_model.append(nn.ReLU())
    my_model.append(nn.ConvTranspose1d(256, 256, kernel_size = 21, stride = 10, padding = 21//2, output_padding = 10 - 1))
    my_model= nn.Sequential(*my_model)


  
    my_model = nn.Sequential(nn.Conv1d(1, 4, kernel_size = 9, stride = 4, padding = 9//2), \
                             nn.Conv1d(4, 4, kernel_size = 7, stride = 3, padding = 7//2), \
                             nn.Conv1d(4, 4, kernel_size = 5, stride = 2, padding = 5//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Conv1d(4, 4, kernel_size = 3, stride = 1, padding = 3//2), \
                             nn.Upsample(scale_factor = 2),\
                             nn.Upsample(scale_factor = 3),\
                             nn.Upsample(scale_factor = 4))
                             #nn.ConvTranspose1d(4, 4, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 2-1), \
                             #nn.ConvTranspose1d(4, 4, kernel_size = 7, stride = 3, padding = 7//2, output_padding = 3-1), \
                             #nn.ConvTranspose1d(4, 1, kernel_size = 9, stride = 4, padding = 9//2, output_padding = 4-1))
                 

    residual_connection = {9 : 0,  8 : 1, 7 : 2}

    my_streamable_model = StreamableModel(buffer_length, my_model, residual_connection)

    test_signal = torch.rand(1, 1, sample_num) 

    output_ref = residual_forward(my_model, residual_connection, test_signal)

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
        full_output = residual_forward(my_model, residual_connection, test_signal[:, :, :buffer_length*(i+1)])

        output_buffer_ref = full_output[:, :, -output_length:]

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
            print("correct output:\n" ,output_buffer_ref[:, 0, :])

                 
        output_full[:, :, output_counter : output_counter + output_length] = output_buffer

        output_counter += full_output_length 
        print("output_counter:", output_counter)
        output_counter -= bad_output_length
        print("___________________________________________________________________")



    print("reference full output:")
    print(output_ref)

    equal = (abs(output_ref - output_full) < 1e-5).all().item()
    if equal:
        print("full output passed")




