# ---
# jupyter:
#   jupytext:
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

# +
#|default_exp layers
#|default_cls_lvl 3
# -

#|export
from __future__ import annotations
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from torch.nn.utils import weight_norm, spectral_norm

#|hide
from nbdev.showdoc import *


# # Layers
# > Custom fastai layers and basic functions to grab them.

# ## Basic manipulations and resize

#|export
def module(*flds, **defaults):
    "Decorator to create an `nn.Module` using `f` as `forward` method"
    pa = [inspect.Parameter(o, inspect.Parameter.POSITIONAL_OR_KEYWORD) for o in flds]
    pb = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=v)
          for k,v in defaults.items()]
    params = pa+pb
    all_flds = [*flds,*defaults.keys()]

    def _f(f):
        class c(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                for i,o in enumerate(args): kwargs[all_flds[i]] = o
                kwargs = merge(defaults,kwargs)
                for k,v in kwargs.items(): setattr(self,k,v)
            __repr__ = basic_repr(all_flds)
            forward = f
        c.__signature__ = inspect.Signature(params)
        c.__name__ = c.__qualname__ = f.__name__
        c.__doc__  = f.__doc__
        return c
    return _f


#|export
@module()
def Identity(self, x):
    "Do nothing at all"
    return x


test_eq(Identity()(1), 1)


#|export
@module('func')
def Lambda(self, x):
    "An easy way to create a pytorch layer for a simple `func`"
    return self.func(x)


def _add2(x): return x+2
tst = Lambda(_add2)
x = torch.randn(10,20)
test_eq(tst(x), x+2)
tst2 = pickle.loads(pickle.dumps(tst))
test_eq(tst2(x), x+2)
tst


#|export
class PartialLambda(Lambda):
    "Layer that applies `partial(func, **kwargs)`"
    def __init__(self, func, **kwargs):
        super().__init__(partial(func, **kwargs))
        self.repr = f'{func.__name__}, {kwargs}'

    def forward(self, x): return self.func(x)
    def __repr__(self): return f'{self.__class__.__name__}({self.repr})'


def test_func(a,b=2): return a+b
tst = PartialLambda(test_func, b=5)
test_eq(tst(x), x+5)


#|export
@module(full=False)
def Flatten(self, x):
    "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
    return x.view(-1) if self.full else x.view(x.size(0), -1)  # Removed cast to Tensorbase


tst = Flatten()
x = torch.randn(10,5,4)
test_eq(tst(x).shape, [10,20])
tst = Flatten(full=True)
test_eq(tst(x).shape, [200])


#|export
@module(tensor_cls=TensorBase)
def ToTensorBase(self, x):
    "Convert x to TensorBase class"
    return self.tensor_cls(x)


ttb = ToTensorBase()
timg = TensorImage(torch.rand(1,3,32,32))
test_eq(type(ttb(timg)), TensorBase)


#|export
class View(Module):
    "Reshape `x` to `size`"
    def __init__(self, *size): self.size = size
    def forward(self, x): return x.view(self.size)


tst = View(10,5,4)
test_eq(tst(x).shape, [10,5,4])


#|export
class ResizeBatch(Module):
    "Reshape `x` to `size`, keeping batch dim the same size"
    def __init__(self, *size): self.size = size
    def forward(self, x): return x.view((x.size(0),) + self.size)


tst = ResizeBatch(5,4)
test_eq(tst(x).shape, [10,5,4])


#|export
@module()
def Debugger(self,x):
    "A module to debug inside a model."
    set_trace()
    return x


#|export
def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


test = tensor([-10.,0.,10.])
assert torch.allclose(sigmoid_range(test, -1,  2), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test, -5, -1), tensor([-5.,-3.,-1.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test,  2,  4), tensor([2.,  3., 4.]), atol=1e-4, rtol=1e-4)


#|export
@module('low','high')
def SigmoidRange(self, x):
    "Sigmoid module with range `(low, high)`"
    return sigmoid_range(x, self.low, self.high)


tst = SigmoidRange(-1, 2)
assert torch.allclose(tst(test), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)


# ## Pooling layers

#|export
class AdaptiveConcatPool1d(Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


#|export
class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


# If the input is `bs x nf x h x h`, the output will be `bs x 2*nf x 1 x 1` if no size is passed or `bs x 2*nf x size x size`

tst = AdaptiveConcatPool2d()
x = torch.randn(10,5,4,4)
test_eq(tst(x).shape, [10,10,1,1])
max1 = torch.max(x,    dim=2, keepdim=True)[0]
maxp = torch.max(max1, dim=3, keepdim=True)[0]
test_eq(tst(x)[:,:5], maxp)
test_eq(tst(x)[:,5:], x.mean(dim=[2,3], keepdim=True))
tst = AdaptiveConcatPool2d(2)
test_eq(tst(x).shape, [10,10,2,2])


#|export
class PoolType: Avg,Max,Cat = 'Avg','Max','Cat'


#|export
def adaptive_pool(pool_type):
    return nn.AdaptiveAvgPool2d if pool_type=='Avg' else nn.AdaptiveMaxPool2d if pool_type=='Max' else AdaptiveConcatPool2d


#|export
class PoolFlatten(nn.Sequential):
    "Combine `nn.AdaptiveAvgPool2d` and `Flatten`."
    def __init__(self, pool_type=PoolType.Avg): super().__init__(adaptive_pool(pool_type)(1), Flatten())


tst = PoolFlatten()
test_eq(tst(x).shape, [10,5])
test_eq(tst(x), x.mean(dim=[2,3]))

# ## BatchNorm layers

#|export
NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')


#|export
def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn


#|export
@delegates(nn.BatchNorm2d)
def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)


#|export
@delegates(nn.InstanceNorm2d)
def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs):
    "InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('InstanceNorm', nf, ndim, zero=norm_type==NormType.InstanceZero, affine=affine, **kwargs)


# `kwargs` are passed to `nn.BatchNorm` and can be `eps`, `momentum`, `affine` and `track_running_stats`.

tst = BatchNorm(15)
assert isinstance(tst, nn.BatchNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = BatchNorm(15, norm_type=NormType.BatchZero)
test_eq(tst.weight, torch.zeros(15))
tst = BatchNorm(15, ndim=1)
assert isinstance(tst, nn.BatchNorm1d)
tst = BatchNorm(15, ndim=3)
assert isinstance(tst, nn.BatchNorm3d)

tst = InstanceNorm(15)
assert isinstance(tst, nn.InstanceNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = InstanceNorm(15, norm_type=NormType.InstanceZero)
test_eq(tst.weight, torch.zeros(15))
tst = InstanceNorm(15, ndim=1)
assert isinstance(tst, nn.InstanceNorm1d)
tst = InstanceNorm(15, ndim=3)
assert isinstance(tst, nn.InstanceNorm3d)

# If `affine` is false the weight should be `None`

test_eq(BatchNorm(15, affine=False).weight, None)
test_eq(InstanceNorm(15, affine=False).weight, None)


#|export
class BatchNorm1dFlat(nn.BatchNorm1d):
    "`nn.BatchNorm1d`, but first flattens leading dimensions"
    def forward(self, x):
        if x.dim()==2: return super().forward(x)
        *f,l = x.shape
        x = x.contiguous().view(-1,l)
        return super().forward(x).view(*f,l)


tst = BatchNorm1dFlat(15)
x = torch.randn(32, 64, 15)
y = tst(x)
mean = x.mean(dim=[0,1])
test_close(tst.running_mean, 0*0.9 + mean*0.1)
var = (x-mean).pow(2).mean(dim=[0,1])
test_close(tst.running_var, 1*0.9 + var*0.1, eps=1e-4)
test_close(y, (x-mean)/torch.sqrt(var+1e-5) * tst.weight + tst.bias, eps=1e-4)


#|export
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


# The `BatchNorm` layer is skipped if `bn=False`, as is the dropout if `p=0.`. Optionally, you can add an activation for after the linear layer with `act`.

# +
tst = LinBnDrop(10, 20)
mods = list(tst.children())
test_eq(len(mods), 2)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Linear)

tst = LinBnDrop(10, 20, p=0.1)
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Dropout)
assert isinstance(mods[2], nn.Linear)

tst = LinBnDrop(10, 20, act=nn.ReLU(), lin_first=True)
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.Linear)
assert isinstance(mods[1], nn.ReLU)
assert isinstance(mods[2], nn.BatchNorm1d)

tst = LinBnDrop(10, 20, bn=False)
mods = list(tst.children())
test_eq(len(mods), 1)
assert isinstance(mods[0], nn.Linear)


# -

# ## Inits

#|export
def sigmoid(input, eps=1e-7):
    "Same as `torch.sigmoid`, plus clamping to `(eps,1-eps)"
    return input.sigmoid().clamp(eps,1-eps)


#|export
def sigmoid_(input, eps=1e-7):
    "Same as `torch.sigmoid_`, plus clamping to `(eps,1-eps)"
    return input.sigmoid_().clamp_(eps,1-eps)


#|export
from torch.nn.init import kaiming_uniform_,uniform_,xavier_uniform_,normal_


#|export
def vleaky_relu(input, inplace=True):
    "`F.leaky_relu` with 0.3 slope"
    return F.leaky_relu(input, negative_slope=0.3, inplace=inplace)


#|export
for o in F.relu,nn.ReLU,F.relu6,nn.ReLU6,F.leaky_relu,nn.LeakyReLU:
    o.__default_init__ = kaiming_uniform_

#|export
for o in F.sigmoid,nn.Sigmoid,F.tanh,nn.Tanh,sigmoid,sigmoid_:
    o.__default_init__ = xavier_uniform_


#|export
def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad(): nested_callable(m, 'bias.fill_')(0.)
    return m


#|export
def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = nested_callable(act_func, '__class__.__default_init__')
        if init == noop: init = getcallable(act_func, '__default_init__')
    if callable(init): init(m.weight)


# ## Convolutions

#|export
def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


#|hide
test_eq(_conv_func(ndim=1),torch.nn.modules.conv.Conv1d)
test_eq(_conv_func(ndim=2),torch.nn.modules.conv.Conv2d)
test_eq(_conv_func(ndim=3),torch.nn.modules.conv.Conv3d)
test_eq(_conv_func(ndim=1, transpose=True),torch.nn.modules.conv.ConvTranspose1d)
test_eq(_conv_func(ndim=2, transpose=True),torch.nn.modules.conv.ConvTranspose2d)
test_eq(_conv_func(ndim=3, transpose=True),torch.nn.modules.conv.ConvTranspose3d)

#|export
defaults.activation=nn.ReLU


#|export
class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    @delegates(nn.Conv2d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=defaults.activation, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)


# The convolution uses `ks` (kernel size) `stride`, `padding` and `bias`. `padding` will default to the appropriate value (`(ks-1)//2` if it's not a transposed conv) and `bias` will default to `True` the `norm_type` is `Spectral` or `Weight`, `False` if it's `Batch` or `BatchZero`. Note that if you don't want any normalization, you should pass `norm_type=None`.
#
# This defines a conv layer with `ndim` (1,2 or 3) that will be a ConvTranspose if `transpose=True`. `act_cls` is the class of the activation function to use (instantiated inside). Pass `act=None` if you don't want an activation function. If you quickly want to change your default activation, you can change the value of `defaults.activation`.
#
# `init` is used to initialize the weights (the bias are initialized to 0) and `xtra` is an optional layer to add at the end.

tst = ConvLayer(16, 32)
mods = list(tst.children())
test_eq(len(mods), 3)
test_eq(mods[1].weight, torch.ones(32))
test_eq(mods[0].padding, (1,1))

x = torch.randn(64, 16, 8, 8)#.cuda()

#Padding is selected to make the shape the same if stride=1
test_eq(tst(x).shape, [64,32,8,8])

#Padding is selected to make the shape half if stride=2
tst = ConvLayer(16, 32, stride=2)
test_eq(tst(x).shape, [64,32,4,4])

#But you can always pass your own padding if you want
tst = ConvLayer(16, 32, padding=0)
test_eq(tst(x).shape, [64,32,6,6])

#No bias by default for Batch NormType
assert mods[0].bias is None
#But can be overridden with `bias=True`
tst = ConvLayer(16, 32, bias=True)
assert first(tst.children()).bias is not None
#For no norm, or spectral/weight, bias is True by default
for t in [None, NormType.Spectral, NormType.Weight]:
    tst = ConvLayer(16, 32, norm_type=t)
    assert first(tst.children()).bias is not None

#Various n_dim/tranpose
tst = ConvLayer(16, 32, ndim=3)
assert isinstance(list(tst.children())[0], nn.Conv3d)
tst = ConvLayer(16, 32, ndim=1, transpose=True)
assert isinstance(list(tst.children())[0], nn.ConvTranspose1d)

#No activation/leaky
tst = ConvLayer(16, 32, ndim=3, act_cls=None)
mods = list(tst.children())
test_eq(len(mods), 2)
tst = ConvLayer(16, 32, ndim=3, act_cls=partial(nn.LeakyReLU, negative_slope=0.1))
mods = list(tst.children())
test_eq(len(mods), 3)
assert isinstance(mods[2], nn.LeakyReLU)


# +
# #export
# def linear(in_features, out_features, bias=True, act_cls=None, init='auto'):
#     "Linear layer followed by optional activation, with optional auto-init"
#     res = nn.Linear(in_features, out_features, bias=bias)
#     if act_cls: act_cls = act_cls()
#     init_linear(res, act_cls, init=init)
#     if act_cls: res = nn.Sequential(res, act_cls)
#     return res

# +
# #export
# @delegates(ConvLayer)
# def conv1d(ni, nf, ks, stride=1, ndim=1, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)

# +
# #export
# @delegates(ConvLayer)
# def conv2d(ni, nf, ks, stride=1, ndim=2, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)

# +
# #export
# @delegates(ConvLayer)
# def conv3d(ni, nf, ks, stride=1, ndim=3, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
# -

#|export
def AdaptiveAvgPool(sz=1, ndim=2):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)


#|export
def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)


#|export
def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


# ## Embeddings

#|export
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


#|export
class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)


# Truncated normal initialization bounds the distribution to avoid large value. For a given standard deviation `std`, the bounds are roughly `-2*std`, `2*std`.

std = 0.02
tst = Embedding(10, 30, std)
assert tst.weight.min() > -2*std
assert tst.weight.max() < 2*std
test_close(tst.weight.mean(), 0, 1e-2)
test_close(tst.weight.std(), std, 0.1)


# ## Self attention

#|export
class SelfAttention(Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


# Self-attention layer as introduced in [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).
#
# Initially, no change is done to the input. This is controlled by a trainable parameter named `gamma` as we return `x + gamma * out`.

tst = SelfAttention(16)
x = torch.randn(32, 16, 8, 8)
test_eq(tst(x),x)

# Then during training `gamma` will probably change since it's a trainable parameter. Let's see what's happening when it gets a nonzero value.

tst.gamma.data.fill_(1.)
y = tst(x)
test_eq(y.shape, [32,16,8,8])

# The attention mechanism requires three matrix multiplications (here represented by 1x1 convs). The multiplications are done on the channel level (the second dimension in our tensor) and we flatten the feature map (which is 8x8 here). As in the paper, we note `f`, `g` and `h` the results of those multiplications.

q,k,v = tst.query[0].weight.data,tst.key[0].weight.data,tst.value[0].weight.data
test_eq([q.shape, k.shape, v.shape], [[2, 16, 1], [2, 16, 1], [16, 16, 1]])
f,g,h = map(lambda m: x.view(32, 16, 64).transpose(1,2) @ m.squeeze().t(), [q,k,v])
test_eq([f.shape, g.shape, h.shape], [[32,64,2], [32,64,2], [32,64,16]])

# The key part of the attention layer is to compute attention weights for each of our location in the feature map (here 8x8 = 64). Those are positive numbers that sum to 1 and tell the model to pay attention to this or that part of the picture. We make the product of `f` and the transpose of `g` (to get something of size bs by 64 by 64) then apply a softmax on the first dimension (to get the positive numbers that sum up to 1). The result can then be multiplied with `h` transposed to get an output of size bs by channels by 64, which we can then be viewed as an output the same size as the original input. 
#
# The final result is then `x + gamma * out` as we saw before.

beta = F.softmax(torch.bmm(f, g.transpose(1,2)), dim=1)
test_eq(beta.shape, [32, 64, 64])
out = torch.bmm(h.transpose(1,2), beta)
test_eq(out.shape, [32, 16, 64])
test_close(y, x + out.view(32, 16, 8, 8), eps=1e-4)


#|export
class PooledSelfAttention2d(Module):
    "Pooled self attention layer for 2d."
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels//2)]
        self.out   = self._conv(n_channels//2, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        n_ftrs = x.shape[2]*x.shape[3]
        f = self.query(x).view(-1, self.n_channels//8, n_ftrs)
        g = F.max_pool2d(self.key(x),   [2,2]).view(-1, self.n_channels//8, n_ftrs//4)
        h = F.max_pool2d(self.value(x), [2,2]).view(-1, self.n_channels//2, n_ftrs//4)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self.out(torch.bmm(h, beta.transpose(1,2)).view(-1, self.n_channels//2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Self-attention layer used in the [Big GAN paper](https://arxiv.org/abs/1809.11096).
#
# It uses the same attention as in `SelfAttention` but adds a max pooling of stride 2 before computing the matrices `g` and `h`: the attention is ported on one of the 2x2 max-pooled window, not the whole feature map. There is also a final matrix product added at the end to the output, before retuning `gamma * out + x`.

#|export
def _conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


#|export
class SimpleSelfAttention(Module):
    def __init__(self, n_in:int, ks=1, sym=False):
        self.sym,self.n_in = sym,n_in
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self,x):
        if self.sym:
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        size = x.size()
        x = x.view(*size[:2],-1)

        convx = self.conv(x)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


# ## PixelShuffle

# PixelShuffle introduced in [this article](https://arxiv.org/pdf/1609.05158.pdf) to avoid checkerboard artifacts when upsampling images. If we want an output with `ch_out` filters, we use a convolution with `ch_out * (r**2)` filters, where `r` is the upsampling factor. Then we reorganize those filters like in the picture below:
#
# <img src="images/pixelshuffle.png" alt="Pixelshuffle" width="800" />

#|export
def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)


# ICNR init was introduced in [this article](https://arxiv.org/abs/1707.02937). It suggests to initialize the convolution that will be used in PixelShuffle so that each of the `r**2` channels get the same weight (so that in the picture above, the 9 colors in a 3 by 3 window are initially the same).
#
# :::{.callout-note}
#
# This is done on the first dimension because PyTorch stores the weights of a convolutional layer in this format: `ch_out x ch_in x ks x ks`.
#
# :::

tst = torch.randn(16*4, 32, 1, 1)
tst = icnr_init(tst)
for i in range(0,16*4,4):
    test_eq(tst[i],tst[i+1])
    test_eq(tst[i],tst[i+2])
    test_eq(tst[i],tst[i+3])


#|export
class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
                  nn.PixelShuffle(scale)]
        if norm_type == NormType.Weight:
            layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
            layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3])**0.5)[:,None,None,None])
        else:
            layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)


# The convolutional layer is initialized with `icnr_init` and passed `act_cls` and `norm_type` (the default of weight normalization seemed to be what's best for super-resolution problems, in our experiments). 
#
# The `blur` option comes from [Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts](https://arxiv.org/abs/1806.02658) where the authors add a little bit of blur to completely get rid of checkerboard artifacts.

psfl = PixelShuffle_ICNR(16)
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])

psfl = PixelShuffle_ICNR(16, norm_type=None)
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])

psfl = PixelShuffle_ICNR(16, norm_type=NormType.Spectral)
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])


# ## Sequential extensions

#|export
def sequential(*args):
    "Create an `nn.Sequential`, wrapping items with `Lambda` if needed"
    if len(args) != 1 or not isinstance(args[0], OrderedDict):
        args = list(args)
        for i,o in enumerate(args):
            if not isinstance(o,nn.Module): args[i] = Lambda(o)
    return nn.Sequential(*args)


#|export
class SequentialEx(Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"
    def __init__(self, *layers): self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
        return res

    def __getitem__(self,i): return self.layers[i]
    def append(self,l):      return self.layers.append(l)
    def extend(self,l):      return self.layers.extend(l)
    def insert(self,i,l):    return self.layers.insert(i,l)


# This is useful to write layers that require to remember the input (like a resnet block) in a sequential way.

#|export
class MergeLayer(Module):
    "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."
    def __init__(self, dense:bool=False): self.dense=dense
    def forward(self, x): return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)


res_block = SequentialEx(ConvLayer(16, 16), ConvLayer(16,16))
res_block.append(MergeLayer()) # just to test append - normally it would be in init params
x = torch.randn(32, 16, 8, 8)
y = res_block(x)
test_eq(y.shape, [32, 16, 8, 8])
test_eq(y, x + res_block[1](res_block[0](x)))

x = TensorBase(torch.randn(32, 16, 8, 8))
y = res_block(x)
test_is(y.orig, None)


# ## Concat

# Equivalent to keras.layers.Concatenate, it will concat the outputs of a ModuleList over a given dimension (default the filter dimension)

#|export 
class Cat(nn.ModuleList):
    "Concatenate layers outputs over a given dim"
    def __init__(self, layers, dim=1):
        self.dim=dim
        super().__init__(layers)
    def forward(self, x): return torch.cat([l(x) for l in self], dim=self.dim)


layers = [ConvLayer(2,4), ConvLayer(2,4), ConvLayer(2,4)] 
x = torch.rand(1,2,8,8) 
cat = Cat(layers) 
test_eq(cat(x).shape, [1,12,8,8]) 
test_eq(cat(x), torch.cat([l(x) for l in layers], dim=1))


# ## Ready-to-go models

#|export
class SimpleCNN(nn.Sequential):
    "Create a simple CNN with `filters`."
    def __init__(self, filters, kernel_szs=None, strides=None, bn=True):
        nl = len(filters)-1
        kernel_szs = ifnone(kernel_szs, [3]*nl)
        strides    = ifnone(strides   , [2]*nl)
        layers = [ConvLayer(filters[i], filters[i+1], kernel_szs[i], stride=strides[i],
                  norm_type=(NormType.Batch if bn and i<nl-1 else None)) for i in range(nl)]
        layers.append(PoolFlatten())
        super().__init__(*layers)


# The model is a succession of convolutional layers from `(filters[0],filters[1])` to `(filters[n-2],filters[n-1])` (if `n` is the length of the `filters` list) followed by a `PoolFlatten`. `kernel_szs` and `strides` defaults to a list of 3s and a list of 2s. If `bn=True` the convolutional layers are successions of conv-relu-batchnorm, otherwise conv-relu.

tst = SimpleCNN([8,16,32])
mods = list(tst.children())
test_eq(len(mods), 3)
test_eq([[m[0].in_channels, m[0].out_channels] for m in mods[:2]], [[8,16], [16,32]])

# Test kernel sizes

tst = SimpleCNN([8,16,32], kernel_szs=[1,3])
mods = list(tst.children())
test_eq([m[0].kernel_size for m in mods[:2]], [(1,1), (3,3)])

# Test strides

tst = SimpleCNN([8,16,32], strides=[1,2])
mods = list(tst.children())
test_eq([m[0].stride for m in mods[:2]], [(1,1),(2,2)])


#|export
class ProdLayer(Module):
    "Merge a shortcut with the result of the module by multiplying them."
    def forward(self, x): return x * x.orig


#|export
inplace_relu = partial(nn.ReLU, inplace=True)


#|export
def SEModule(ch, reduction, act_cls=defaults.activation):
    nf = math.ceil(ch//reduction/8)*8
    return SequentialEx(nn.AdaptiveAvgPool2d(1),
                        ConvLayer(ch, nf, ks=1, norm_type=None, act_cls=act_cls),
                        ConvLayer(nf, ch, ks=1, norm_type=None, act_cls=nn.Sigmoid),
                        ProdLayer())


#|export
class ResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=defaults.activation, ndim=2, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
                     ConvLayer(nh2,  nf, ks, groups=g2, **k1)
        ] if expansion == 1 else [
                     ConvLayer(ni,  nh1, 1, **k0),
                     ConvLayer(nh1, nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))


# This is a resnet block (normal or bottleneck depending on `expansion`, 1 for the normal block and 4 for the traditional bottleneck) that implements the tweaks from [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187). In particular, the last batchnorm layer (if that is the selected `norm_type`) is initialized with a weight (or gamma) of zero to facilitate the flow from the beginning to the end of the network. It also implements optional [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) and grouped convs for [ResNeXT](https://arxiv.org/abs/1611.05431) and similar models (use `dw=True` for depthwise convs).
#
# The `kwargs` are passed to `ConvLayer` along with `norm_type`.

#|export
def SEBlock(expansion, ni, nf, groups=1, reduction=16, stride=1, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh1=nf*2, nh2=nf*expansion, **kwargs)


#|export
def SEResNeXtBlock(expansion, ni, nf, groups=32, reduction=16, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh2=w, **kwargs)


#|export
def SeparableBlock(expansion, ni, nf, reduction=16, stride=1, base_width=4, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, reduction=reduction, nh2=nf*2, dw=True, **kwargs)


# ## Time Distributed Layer

# Equivalent to Keras `TimeDistributed` Layer, enables computing pytorch `Module` over an axis.

#|export
def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(torch.stack([t[i] for t in tuples], dim=stack_dim) for i in range_of(tuples[0]))


#|export
class TimeDistributed(Module):
    "Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time." 
    def __init__(self, module, low_mem=False, tdim=1):
        store_attr()
        
    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            #only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.view(bs*seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)
    
    def low_mem_forward(self, *tensors, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)
    
    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len,*out.shape[1:])
    
    def __repr__(self):
        return f'TimeDistributed({self.module})'


bs, seq_len = 2, 5
x, y = torch.rand(bs,seq_len,3,2,2), torch.rand(bs,seq_len,3,2,2)

tconv = TimeDistributed(nn.Conv2d(3,4,1))
test_eq(tconv(x).shape, (2,5,4,2,2))
tconv.low_mem=True
test_eq(tconv(x).shape, (2,5,4,2,2))


class Mod(Module):
    def __init__(self):
        self.conv = nn.Conv2d(3,4,1)
    def forward(self, x, y):
        return self.conv(x) + self.conv(y)
tmod = TimeDistributed(Mod())

out = tmod(x,y)
test_eq(out.shape, (2,5,4,2,2))
tmod.low_mem=True
out_low_mem = tmod(x,y)
test_eq(out_low_mem.shape, (2,5,4,2,2))
test_eq(out, out_low_mem)


class Mod2(Module):
    def __init__(self):
        self.conv = nn.Conv2d(3,4,1)
    def forward(self, x, y):
        return self.conv(x), self.conv(y)
tmod2 = TimeDistributed(Mod2())

out = tmod2(x,y)
test_eq(len(out), 2)
test_eq(out[0].shape, (2,5,4,2,2))
tmod2.low_mem=True
out_low_mem = tmod2(x,y)
test_eq(out_low_mem[0].shape, (2,5,4,2,2))
test_eq(out, out_low_mem)

show_doc(TimeDistributed)

# This module is equivalent to [Keras TimeDistributed Layer](https://keras.io/api/layers/recurrent_layers/time_distributed/). This wrapper allows to apply a layer to every temporal slice of an input. By default it is assumed the time axis (`tdim`) is the 1st one (the one after the batch size). A typical usage would be to encode a sequence of images using an image encoder.
#
# The `forward` function of `TimeDistributed` supports `*args` and `**kkwargs` but only `args` will be split and passed to the underlying module independently for each timestep, `kwargs` will be passed as they are. This is useful when you have module that take multiple arguments as inputs, this way, you can put all tensors you need spliting as `args` and other arguments that don't need split as `kwargs`.
#
# > This module is heavy on memory, as it will try to pass mutiple timesteps at the same time on the batch dimension, if you get out of memorey errors, try first reducing your batch size by the number of timesteps.

from fastai.vision.all import *

encoder = create_body(resnet18())

# A resnet18 will encode a feature map of 512 channels. Height and Width will be divided by 32.

time_resnet = TimeDistributed(encoder)

# a synthetic batch of 2 image-sequences of lenght 5. `(bs, seq_len, ch, w, h)`

image_sequence = torch.rand(2, 5, 3, 64, 64)

time_resnet(image_sequence).shape

# This way, one can encode a sequence of images on feature space.
# There is also a `low_mem_forward` that will pass images one at a time to reduce GPU memory consumption.

time_resnet.low_mem_forward(image_sequence).shape

# ## Swish and Mish

#|export
from torch.jit import script


# +
#|export
@script
def _swish_jit_fwd(x): return x.mul(torch.sigmoid(x))

@script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))

class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _swish_jit_bwd(x, grad_output)


# -

#|export
def swish(x, inplace=False): F.silu(x, inplace=inplace)


#|export
class SwishJit(Module):
    def forward(self, x): return _SwishJitAutoFn.apply(x)


# +
#|export
@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))

@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)


# -

#|export
def mish(x, inplace=False): return F.mish(x, inplace=inplace)


#|export
class MishJit(Module):
    def forward(self, x): return MishJitAutoFn.apply(x)


#|export
Mish = nn.Mish
Swish = nn.SiLU

#|export
for o in swish,Swish,SwishJit,mish,Mish,MishJit: o.__default_init__ = kaiming_uniform_


# ## Helper functions for submodules

# It's easy to get the list of all parameters of a given model. For when you want all submodules (like linear/conv layers) without forgetting lone parameters, the following class wraps those in fake modules.

#|export
class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p): self.val = p
    def forward(self, x): return x


#|export
def children_and_parameters(m):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children


# +
class TstModule(Module):
    def __init__(self): self.a,self.lin = nn.Parameter(torch.randn(1)),nn.Linear(5,10)

tst = TstModule()
children = children_and_parameters(tst)
test_eq(len(children), 2)
test_eq(children[0], tst.lin)
assert isinstance(children[1], ParameterModule)
test_eq(children[1].val, tst.a)


# -

#|export
def has_children(m):
    try: next(m.children())
    except StopIteration: return False
    return True


class A(Module): pass
assert not has_children(A())
assert has_children(TstModule())


#|export
def flatten_model(m):
    "Return the list of all submodules and parameters of `m`"
    return sum(map(flatten_model,children_and_parameters(m)),[]) if has_children(m) else [m]


tst = nn.Sequential(TstModule(), TstModule())
children = flatten_model(tst)
test_eq(len(children), 4)
assert isinstance(children[1], ParameterModule)
assert isinstance(children[3], ParameterModule)


#|export
class NoneReduce():
    "A context manager to evaluate `loss_func` with none reduce."
    def __init__(self, loss_func): self.loss_func,self.old_red = loss_func,None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = self.loss_func.reduction
            self.loss_func.reduction = 'none'
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: self.loss_func.reduction = self.old_red


# +
x,y = torch.randn(5),torch.randn(5)
loss_fn = nn.MSELoss()
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn.reduction, 'mean')

loss_fn = F.mse_loss
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn, F.mse_loss)


# -

#|export
def in_channels(m):
    "Return the shape of the first weight layer in `m`."
    try: return next(l.weight.shape[1] for l in flatten_model(m) if nested_attr(l,'weight.ndim',-1)==4)
    except StopIteration as e: e.args = ["No weight layer"]; raise


test_eq(in_channels(nn.Sequential(nn.Conv2d(5,4,3), nn.Conv2d(4,3,3))), 5)
test_eq(in_channels(nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(BatchNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4, affine=False), nn.Conv2d(4,3,3))), 4)
test_fail(lambda : in_channels(nn.Sequential(nn.AvgPool2d(4))))

# ## Export -

#|hide
from nbdev import *
nbdev_export()


