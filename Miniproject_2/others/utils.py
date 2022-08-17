from torch import empty
from torch.nn.functional import fold, unfold
import math
from torch import device, cuda, save, load, Tensor


class Module(object):

    #this module will be inherited by all the classes
    #contains the basic functions needed

    def __init__(self):
        #sets the device that all the classes will use. uses cuda if available
        self.device = device("cuda") if cuda.is_available() else device("cpu")

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def __call__(self, input):
        #calls the forward function so that each class is subscriptable
        return self.forward(input)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        #initialises all the parameters of the model, including the weights
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        k = groups / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = self._RandomTensor((out_channels, int(in_channels / groups), self.kernel_size[0], self.kernel_size[1]), -math.sqrt(k),
                              math.sqrt(k)).to(self.device)
        self.dL_dw = empty(self.weight.size()).to(self.device)
        self.bias = self._RandomTensor(out_channels, -math.sqrt(k), math.sqrt(k)).to(self.device)
        self.dL_db = empty(self.bias.size()).to(self.device)
        self.is_bias = bias

    @staticmethod
    def _RandomTensor(size, low, up):
        #method that creates a tensor of random numbers with a predefined size
        MatToFill = empty(size)
        return MatToFill.uniform_(low, up)

    def forward(self, input):
        #performs forward pass
        self.input = input.to(self.device)
        output_size = (int((input.shape[2] - self.kernel_size[0] + 2 * self.padding) / self.stride + 1),
                       int((input.shape[3] - self.kernel_size[1] + 2 * self.padding) / self.stride + 1))

        unfolded = unfold(self.input, self.kernel_size, dilation=self.dilation, padding=self.padding,
                          stride=self.stride)

        conv_output = self.weight.view(self.weight.size(0), -1)

        if self.is_bias:
            conv_output = conv_output.matmul(unfolded) + self.bias.view(1, -1, 1)

        else:
            conv_output = conv_output.matmul(unfolded)

        folded = fold(conv_output,
                      output_size=output_size,
                      kernel_size=(1, 1))

        self.dL_dZ = folded

        return folded

    def backward(self, *gradwrtoutput):
        #performs backward pass
        dL_dZ = gradwrtoutput[0] if gradwrtoutput else self.dL_dZ

        unfolded = unfold(self.input, self.kernel_size, dilation=self.dilation, padding=self.padding,
                          stride=self.stride)
        conv_output = dL_dZ.view(dL_dZ.size(0), dL_dZ.size(1), -1).matmul(unfolded.transpose(1, 2))
        self.dL_dw.data = conv_output.sum(0).reshape(self.weight.shape)

        if self.is_bias:
            self.dL_db.data = dL_dZ.sum(dim=[0, 2, 3])  # should be output channel

        batch, c, h, w = self.input.shape
        output_size = (
            batch, dL_dZ.size(1), (h - 1) * self.stride - 2 * self.padding + self.kernel_size[0] + h % self.stride,
            (w - 1) * self.stride - 2 * self.padding + self.kernel_size[1] + w % self.stride)
        pad = (int((output_size[2] - dL_dZ.size(2)) / 2) + dL_dZ.size(2) % 2,
               int((output_size[3] - dL_dZ.size(3)) / 2) + dL_dZ.size(3) % 2)
        w_rotated = self.weight.rot90(2, [2, 3]).transpose(0, 1)
        unfolded = unfold(dL_dZ, self.kernel_size, stride=self.stride, padding=pad, dilation=self.dilation)
        conv_output = w_rotated.reshape(self.input.shape[1], unfolded.shape[1]).matmul(unfolded)

        dL_dX = fold(conv_output, (self.input.shape[2], self.input.shape[3]), kernel_size=(1, 1))
        return dL_dX

    def param(self):
        return [[self.bias, self.dL_db], [self.weight, self.dL_dw]] if self.is_bias else [[self.weight, self.dL_dw]]


class Upsampling(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True):
        #initialises all the parameters of the model, including the weights
        #inherits the parameters and functions from the Conv2 class
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale_factor = stride * 2
        self.is_bias = bias

    def forward(self, input):
        #performs forward pass
        self.input = input
        self.expanded = input.repeat_interleave(4, dim=3).repeat_interleave(4, dim=2)

        self.output = super().forward(self.expanded)
        return self.output

    def backward(self, *gradwrtoutput):
        #performs backward pass
        dL_dZ = gradwrtoutput[0] if gradwrtoutput else self.output
        self.dL_dX = super().backward(dL_dZ)

        dl_dlX = []
        for i in range(self.scale_factor):
            for j in range(self.scale_factor):
                dl_dlX.append(self.dL_dX[:, :, i::self.scale_factor, j::self.scale_factor])

        return sum(dl_dlX)

    def param(self):
        return [[self.bias, self.dL_db], [self.weight, self.dL_dw]] if self.is_bias else [[self.weight, self.dL_dw]]


class SGD(Module):
    def __init__(self, parameters, lr):
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def step(self):
        #updates parameters
        for p, dp in self.parameters:
            p.data -= self.lr * dp

    def zero_grad(self):
        #puts all the gradients to 0
        for p, dp in self.parameters:
            dp.data = empty(dp.shape)


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.parameters = []

    def forward(self, input):
        #performs forward pass
        self.input = input
        return (input > 0) * input

    def backward(self, *gradwrtoutput):
        #performs backward pass
        gradwrtoutput = gradwrtoutput[0]
        return ((self.input > 0) * gradwrtoutput).float()


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.parameters = []

    def forward(self, input):
        #performs forward pass
        self.input = input
        return 1 / (1 + (-input).exp())

    def backward(self, *gradwrtoutput):
        #performs backward pass
        return ((-self.input).exp() / (((-self.input).exp() + 1) ** (2))).mul(gradwrtoutput[0])


class MSE(Module):
    def __init__(self):
        super().__init__()
        self.parameters = []

    def forward(self, input, target):
        #performs forward pass
        self.input = input
        self.target = target.to(self.device)
        SE = (input.sub(self.target)) ** 2
        return SE.mean()

    def backward(self):
        #performs backward pass
        tot = self.input.shape[0] * self.input.shape[1] * self.input.shape[2] * self.input.shape[3]
        return 2 / tot * (self.input.sub(self.target))

    def __call__(self, input, target):
        #calls forward with input and target
        return self.forward(input, target)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.parameters = []

    def forward(self, input):
        #performs forward sequentially on the models it contains
        self.output = input

        for arg in self.args:
            self.output = arg(self.output)
        return self.output

    def backward(self, *gradwrtoutput):
        #performs backward sequentially on the models it contains
        gradwrtoutput = gradwrtoutput[0] if gradwrtoutput else self.output

        for arg in reversed(self.args):
            gradwrtoutput = arg.backward(gradwrtoutput)
        return gradwrtoutput

    def param(self):
        #accumulates all the parameters from all the models it contains
        for idx, arg in enumerate(self.args):
            for idx2, p in enumerate(arg.param()):
                self.parameters.append(p)
        return self.parameters