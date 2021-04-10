import torch
import torch.nn as nn
import threading

'''
Customized 1D convolutional layer in PyTorch
'''
class MyConv1D(nn.Module):
    '''
    @params:
    in_channels: input signal channel number 
    out_channels: output signal channel number
    kernel_size: length of convolution kernel
    stride: number of sample shifts over input signal
    bias_flag: true to allow bias tensor, false otherwise
    '''
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride = 1, bias_flag = True, input_buffer_size = 0):
        super().__init__()
        self.in_channels, self.out_channels= in_channels, out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias_flag = bias_flag
        self.lock = threading.Lock()

        # define 1D convolution kernel, shape = (out_channels, in_channels, kernel_size)
        kernel = torch.Tensor(out_channels, in_channels, kernel_size)
        self.kernel = nn.Parameter(kernel)  

        # define bias, shape = (out_channels)
        if bias_flag:
            bias = torch.Tensor(out_channels)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        # initialize kernel and bias
        nn.init.uniform_(self.kernel) 
        if bias_flag:
            nn.init.uniform_(self.bias)  

        # create a buffer for input 
        if input_buffer_size != 0:
            assert input_buffer_size > kernel_size, "tile size must be larger than kernel size"
            # buffer to store a tile of input
            self.input_buffer_size = input_buffer_size
            self.input_buffer = torch.zeros((in_channels, input_buffer_size)) 
            # output length of the input_buffer_size tile
            self.tile_out_len = int((input_buffer_size - kernel_size + 1 - 1)/stride + 1)
            # count the number of elements not yet compute convolution
            self.tile_counter = 0 
            # perform a tiled convolution when tile_counter == tile_counter_max                            
            self.tile_counter_max = input_buffer_size - kernel_size + 1
            # tile_full is true if input_buffer is full
            self.tile_full = False
        # not use tiles 
        else:
            self.input_buffer_size = 0

    '''
    perform 1D convolution, input x has self.in_channels channels and output y has self.out_channels channels
    @params:
    x: 2D tensor: x.shape = (self.in_channels, x_len)
    @return:
    y: 2D tensor: y.shape = (self.out_channels, y_len)
                y_len = floor((x_len - self.kernel_size + 1 - 1)/self.stride + 1)
    '''
    def conv_1d(self,x): 
        assert x.shape[0] == self.in_channels, "input and kernel channels mismatch!"
        x_len = x.shape[1]
        y_len = int((x_len - self.kernel_size + 1 - 1)/self.stride + 1)
        
        #x_strided.shape = (self.in_channels, y_len, kernel_size)
        x_strided  = torch.as_strided(x, (self.in_channels, y_len, self.kernel_size),(x_len, self.stride, 1))

        #tensordot: sum over elements in self.in_channels and kernel_size dimensions 
        y = torch.tensordot(x_strided, self.kernel, dims=([0,2],[1,2]))
        if self.bias_flag != False:
            y += self.bias
        return y.T

    '''
    perform 1D convolution with streaming
    @return:
    y: 2D tensor: y.shape = (self.out_channels, y_len)
                y_len = floor((x_len - self.kernel_size + 1 - 1)/self.stride + 1)
    '''
    def conv_1d_streaming(self, x_len): 
        y_len = int((x_len - self.kernel_size + 1 - 1)/self.stride + 1)
        y = torch.zeros((self.out_channels, y_len))
        y_start = 0

        while y_start < y_len:
            self.lock.acquire()
            try:
                if ((not self.tile_full) and (self.tile_counter == self.input_buffer_size))\
                    or (self.tile_full and self.tile_counter == self.tile_counter_max):

                    self.tile_full = True

                    y_end = y_start + self.tile_out_len
                    if y_end > y_len:
                        y_end = y_len

                    # perform partial 1D convolution
                    tile_strided  = torch.as_strided(self.input_buffer, (self.in_channels, self.tile_out_len, self.kernel_size),(self.input_buffer_size, self.stride, 1))
                    if self.bias_flag:
                        y[:, y_start:y_end] = ((torch.tensordot(tile_strided, self.kernel, dims=([0,2],[1,2])) + self.bias).T)[:, 0: (y_end-y_start)]
                    else:
                        y[:, y_start:y_end] = ((torch.tensordot(tile_strided, self.kernel, dims=([0,2],[1,2]))).T)[:, 0: (y_end-y_start)]

                    print(y[:, y_start:y_end])

                    y_start = y_end
                    self.tile_counter = 0
            finally:
                self.lock.release()

            
        return y

    def forward(self, x = None, x_len = 0):
        # normal 1D convolution
        if self.input_buffer_size == 0:
            return self.conv_1d(x)
        # streaming 1D convolution
        else:
            return self.conv_1d_streaming(x_len)

    # manually set kernal for testing
    def set_kernel(self, kernel):
        assert kernel.shape == torch.Size([self.out_channels, self.in_channels, self.kernel_size]), "input kernel shape mismatch!"
        self.kernel = nn.Parameter(kernel) 

    # manually set bias for testing
    def set_bias(self, bias):
        assert bias.shape == torch.Size([self.out_channels]), "input bias shape mismatch!"
        self.bias = nn.Parameter(bias) 

'''
generate real-time random samples
@params:
conv1Dlayer: self.input_buffer.shape = (self.in_channels, self.input_buffer_size)
delta_t: generation period in ms
rand_samples: previously generated samples for testing
'''
def generate_samples(conv1Dlayer:MyConv1D, delta_t:int, rand_samples:torch.tensor):
    channels = rand_samples.shape[0]
    length = rand_samples.shape[1]
    assert channels == conv1Dlayer.input_buffer.shape[0], "generated samples have wrong dimension"
    counter = 0
    samples = rand_samples
    padding_length = (conv1Dlayer.tile_counter_max - (length - conv1Dlayer.input_buffer_size) % conv1Dlayer.tile_counter_max)%conv1Dlayer.tile_counter_max
    # zero padding if the last tile cannot be fully filled 
    if padding_length != 0:
        samples = torch.cat((samples, torch.zeros((channels, padding_length))), dim = 1)

    while counter < length + padding_length:
        #new_sample = torch.randn(channels)
        # left shift buffer by one and welcome new sample 
        conv1Dlayer.lock.acquire()
        try:
            conv1Dlayer.input_buffer = torch.roll(conv1Dlayer.input_buffer, -1, 1)
            conv1Dlayer.input_buffer[:, -1] = samples[:, counter] #new_sample
            conv1Dlayer.tile_counter += 1
        finally:
            conv1Dlayer.lock.release()

        counter += 1

        time.sleep(delta_t)



'''
for debugging 
'''

'''
perform 1D 1 channel convolution
@params:
x: tensor: x.shape = (x_len)
kernel: tensor: kernel.shape = (kernel_size)
bias: tensor: bias.shape = (1)
stride: integer
return:
y: tensor: y.shape = (y_len)
                    y_len = floor((x_len - kernel_size + 1 - 1)/stride + 1)
'''
def conv_1d_1C(x, kernel, bias = None, stride = 1): 
    x_len = len(x)
    kernel_size = len(kernel)
    y_len = int((x_len - kernel_size + 1 - 1)/stride + 1)
    x_strided = torch.as_strided(x, (y_len, kernel_size),(stride, 1))
    y = torch.matmul(x_strided, kernel)
    if bias != None:
        y += bias
    return y

'''
perform 1D convolution, input x has one channel, but output y has y_ch channels
@params:
x: 1D tensor: x.shape = (x_len)
kernel: 2D tensor: kernel.shape = (y_ch, kernel_size)
bias: 1D tensor: bias.shape = (y_ch)
stride: integer
return:
y: 2D tensor: y.shape = (y_ch, y_len)
                y_len = floor((x_len - kernel_size + 1 - 1)/stride + 1)
'''
def conv_1d_MC(x, kernel, bias = None, stride = 1): 
    x_len = len(x)
    kernel_size = kernel.shape[1]
    y_len = int((x_len - kernel_size + 1 - 1)/stride + 1)
    #x_strided.shape = (y_len, kernel_size)
    x_strided = torch.as_strided(x, (y_len, kernel_size),(stride, 1))
    y = torch.matmul(x_strided, kernel.T)
    if bias != None:
        y += bias
    return y.T

'''
perform 1D convolution, input x has x_ch channels and output y has y_ch channels
@params:
x: 2D tensor: x.shape = (x_ch, x_len)
kernel: 2D tensor: kernel.shape = (y_ch, x_ch, kernel_size)
bias: 1D tensor: bias.shape = (y_ch)
stride: integer
return:
y: 2D tensor: y.shape = (y_ch, y_len)
                y_len = floor((x_len - kernel_size + 1 - 1)/stride + 1)
'''
def conv_1d(x, kernel, bias = None, stride = 1): 
    x_ch  = x.shape[0]
    x_len = x.shape[1]
    kernel_size = kernel.shape[2]
    y_len = int((x_len - kernel_size + 1 - 1)/stride + 1)
    
    #x_strided.shape = (x_ch, y_len, kernel_size)
    x_strided  = torch.as_strided(x, (x_ch, y_len, kernel_size),(x_len, stride, 1))

    #tensordot: sum over elements in x_ch and kernel_size dimensions 
    y = torch.tensordot(x_strided, kernel, dims=([0,2],[1,2]))
    if bias != None:
        y += bias
    return y.T



# testing
if __name__ == '__main__':
    stride = 2
    kernel_size = 5
    in_channels = 4
    out_channels = 3
    x_len = 50
    input_buffer_size = 10

    '''
    test1: single output channel, stride = 1
    '''
    print("---------test1---------")
    # number of inputs * input channel * length
    x = torch.Tensor([[[1,2,3,4,5,6,7,8,9,10]]])
    # output channel * input channel/(groups = 1) * kernel size
    filter = torch.Tensor([[[1,2,1]]])
    y1 = F.conv1d(x, filter, stride = 1)
    y2 = conv_1d_1C(x[0,0], filter[0,0], 0, 1)
    print(y1)
    print(y2)
    '''
    test2: single output channel, stride = 2
    '''
    print("---------test2---------")
    x = torch.randn(1, 1, 47)
    filter = torch.randn(1, 1, 5)
    y1 = F.conv1d(x, filter, bias = torch.FloatTensor([2]), stride = 4)
    y2 = conv_1d_1C(x[0,0], filter[0,0], bias = torch.FloatTensor([2]), stride = 4)
    print(y1)
    print(y2)
    '''
    test3: multiple output channels
    '''
    print("---------test3---------")
    x = torch.randn(1, 1, x_len)
    filter = torch.randn(3, 1, out_channels)
    bias = torch.randn(out_channels)
    y1 = F.conv1d(x, filter, bias = bias, stride = stride)
    y2 = conv_1d_MC(x[0,0], filter[:, 0, :], bias = bias, stride = stride)
    print(y1)
    print(y2)
    '''
    test4: multiple input and output channels
    '''
    print("---------test4---------")
    # number of inputs * input channel * length
    x = torch.randn(1, in_channels, x_len)
    # output channel * input channel * kernel size
    filter = torch.randn(out_channels, in_channels, kernel_size)
    bias = torch.randn(out_channels)
    y1 = F.conv1d(x, filter, bias = bias, stride = stride)
    y2 = conv_1d(x[0], filter, bias = bias, stride = stride)
    print(y1)
    print(y2)
    '''
    test5: incorporate into a network layer
    '''
    print("---------test5---------")
    # number of inputs * input channel * length
    x = torch.randn(1, in_channels, x_len)
    # output channel * input channel * kernel size
    filter = torch.randn(out_channels, in_channels, kernel_size)
    bias = torch.randn(out_channels)
    y1 = F.conv1d(x, filter, bias = bias, stride = stride)

    conv1Dlayer = MyConv1D(in_channels, out_channels, kernel_size, stride, True)
    conv1Dlayer.set_kernel(filter)
    conv1Dlayer.set_bias(bias)
    y2 = conv1Dlayer(x[0])
    '''
    for name, param in conv1Dlayer.named_parameters():
        print(name, param)
    '''
    print(y1)
    print(y2)


    '''
    test6: mimic real-time audio and streaming convolution
    '''
    print("---------test6---------")
    conv1Dlayer = MyConv1D(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias_flag = True, input_buffer_size = input_buffer_size)
    conv1Dlayer.set_kernel(filter)
    conv1Dlayer.set_bias(bias)

    # generate audio and streaming 1D convolution 
    real_time_audio = threading.Thread(target=generate_samples, args=(conv1Dlayer, 0.0001, x[0]))
    real_time_audio.start()
    y = conv1Dlayer(x_len = x_len)
    print(y - y2)