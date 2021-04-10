import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_dev import conv_1d_1C, conv_1d_MC, conv_1d, MyConv1D, generate_samples

'''
streaming 1D convolutional layer in PyTorch
'''
class Conv1D_streaming(nn.Module):
    '''
    @params:
    in_channels: input signal channel number 
    out_channels: output signal channel number
    kernel_size: length of convolution kernel
    stride: number of sample shifts over input signal
    max_output_size: the maximum number of samples the layer computes and stores
    bias_flag: true to allow bias tensor, false otherwise
    stride: number of sample shifts over input signal
    '''
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, max_output_size:int, bias_flag = True, stride = 1):
        super().__init__()
        self.in_channels, self.out_channels= in_channels, out_channels
        assert stride < kernel_size, "stride must be smaller than kernel_size"
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias_flag = bias_flag

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

        # buffer to store a tile of input
        self.input_buffer_size = self.kernel_size - 1
        self.input_buffer = torch.zeros((in_channels, self.input_buffer_size)) 
        self.input_counter = 0

        # buffer to store convolution output
        self.max_output_size = max_output_size
        self.output_buffer = torch.zeros((out_channels, max_output_size)) 
        self.output_counter = 0
 
    '''
    perform streaming 1D convolution on x and stores the result in self.output_buffer
    @params:
    x: new input samples: x.shape = (self.in_channels, x_len)
    '''
    def conv_1d_streaming(self, x): 
        assert x.shape[0] == self.in_channels, "input samples x has the wrong number of channels"
        if self.output_counter >= self.max_output_size:
            print("REACH OUTPUT NUMBER MAXIMUM")
            return -1

        # concatenate new input samples with previous samples in the input buffer
        prev_len = self.input_counter - self.output_counter * self.stride   
        self.input_counter += x.shape[1]
        x_len = prev_len + x.shape[1]

        if prev_len != 0:
            x = torch.cat((self.input_buffer[:, -prev_len:], x), 1)
        else:
            x = x.clone().detach()
    
        # input buffer []

        # 0 |   |
        # 4 |   |
        # 8 |   |
        # 0 |        |     |            | 
        # 20|        |     |            |
        # 40|        |     |            |   

        # x_plus[:, 4: 9]
        #

        # where to put results in self.output_buffer
        y_len = max(int((x_len - self.kernel_size)/self.stride + 1), 0)
        y_start = self.output_counter
        y_end = min(y_start + y_len, self.max_output_size)

        # store latest input samples in self.input_buffer
        push_len = self.kernel_size - (y_end * self.stride + self.kernel_size - self.input_counter)
        self.input_buffer[:, -push_len:] = x[:, -push_len:]

        # not ready
        if y_len == 0:
            return

        # perform partial 1D convolution
        # NOTE: in order for torch.as_strided to work properly, each channel of x must seperate exactly x_len in the memory!
        x_strided  = torch.as_strided(x, (self.in_channels, y_len, self.kernel_size),(x_len, self.stride, 1))

        if self.bias_flag:
            self.output_buffer[:, y_start:y_end] = ((torch.tensordot(x_strided, self.kernel, dims=([0,2],[1,2])) + self.bias).T)[:, 0: (y_end-y_start)]
        else:
            self.output_buffer[:, y_start:y_end] = ((torch.tensordot(x_strided, self.kernel, dims=([0,2],[1,2]))).T)[:, 0: (y_end-y_start)]

        self.output_counter = y_end          

        return self.output_buffer[:, y_start:y_end]

    def forward(self, x):
        return self.conv_1d_streaming(x)

    # manually set kernal for testing
    def set_kernel(self, kernel):
        assert kernel.shape == torch.Size([self.out_channels, self.in_channels, self.kernel_size]), "input kernel shape mismatch!"
        self.kernel = nn.Parameter(kernel) 

    # manually set bias for testing
    def set_bias(self, bias):
        assert bias.shape == torch.Size([self.out_channels]), "input bias shape mismatch!"
        self.bias = nn.Parameter(bias) 

# testing
if __name__ == '__main__':
    stride = 2
    kernel_size = 5
    in_channels = 3
    out_channels = 4
    x_len = 500
    max_output_size = 500
    samples_per_call = 100


    mid_channels1 = 4
    mid_channels2 = 5
    kernel_size2 = 7
    kernel_size3 = 3
    stride2 = 3
    stride3 = 1

#################################################################################################################################
    '''
    test7: streaming convolution layer
    '''
    print("---------test7---------")
    conv1Dlayer = Conv1D_streaming(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, max_output_size = max_output_size, bias_flag = True, stride = stride)

    kernel = torch.randn(out_channels, in_channels, kernel_size)
    bias = torch.randn(out_channels)
    conv1Dlayer.set_kernel(kernel)
    conv1Dlayer.set_bias(bias)

    x = torch.randn(1, in_channels, x_len)
    y_ref = F.conv1d(x, kernel, bias = bias, stride = stride)[0]

    for i in range(0, x_len, samples_per_call):
        conv1Dlayer(x[0, :, i : i + samples_per_call].clone().detach())

    y = conv1Dlayer.output_buffer
 
    #print(y_ref)
    #print(y[:, 0:y_ref.shape[1]])
    print("total number of mismatch for y: ", torch.sum(torch.abs(y_ref - y[:, 0:y_ref.shape[1]]) > 1e-4))


    '''
    test8: multi streaming convolution layers
    '''
    print("---------test8---------")
    conv1Dlayer1 = Conv1D_streaming(in_channels = in_channels, out_channels = mid_channels1, kernel_size = kernel_size, max_output_size = max_output_size, bias_flag = True, stride = stride)
    conv1Dlayer2 = Conv1D_streaming(in_channels = mid_channels1, out_channels = mid_channels2, kernel_size = kernel_size2, max_output_size = max_output_size, bias_flag = True, stride = stride2)
    conv1Dlayer3 = Conv1D_streaming(in_channels = mid_channels2, out_channels = out_channels, kernel_size = kernel_size3, max_output_size = max_output_size, bias_flag = True, stride = stride3)

    kernel1 = torch.randn(mid_channels1, in_channels, kernel_size)
    bias1 = torch.randn(mid_channels1)
    kernel2 = torch.randn(mid_channels2, mid_channels1, kernel_size2)
    bias2 = torch.randn(mid_channels2)
    kernel3 = torch.randn(out_channels, mid_channels2, kernel_size3)
    bias3 = torch.randn(out_channels)

    conv1Dlayer1.set_kernel(kernel1)
    conv1Dlayer1.set_bias(bias1)
    conv1Dlayer2.set_kernel(kernel2)
    conv1Dlayer2.set_bias(bias2)
    conv1Dlayer3.set_kernel(kernel3)
    conv1Dlayer3.set_bias(bias3)

    x = torch.randn(1, in_channels, x_len)
    e1_ref = F.conv1d(x, kernel1, bias = bias1, stride = stride)
    e2_ref = F.conv1d(e1_ref, kernel2, bias = bias2, stride = stride2)
    y_ref  = F.conv1d(e2_ref, kernel3, bias = bias3, stride = stride3)

    e1_ref = e1_ref[0]
    e2_ref = e2_ref[0]
    y_ref  = y_ref[0]

    #steamable_model
    #init(num_samples for first layer, ):
    #   compute dependency
    #   initialize layers
    #   nn.Sequential
    #    
    #   
    #  8, kernel_size = 3, stride = 3
    #  2 

    for i in range(0, x_len, samples_per_call):
        conv1Dlayer3(conv1Dlayer2(conv1Dlayer1(x[0, :, i : i + samples_per_call])))
        
    e1 = conv1Dlayer1.output_buffer
    e2 = conv1Dlayer2.output_buffer
    y  = conv1Dlayer3.output_buffer

    print("total number of mismatch for e1: ", torch.sum(torch.abs(e1_ref - e1[:, 0:e1_ref.shape[1]]) > 1e-4))
    print("total number of mismatch for e2: ", torch.sum(torch.abs(e2_ref - e2[:, 0:e2_ref.shape[1]]) > 1e-4))
    print("total number of mismatch for y: ",  torch.sum(torch.abs(y_ref - y[:, 0:y_ref.shape[1]]) > 1e-4))





