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
#|export
from __future__ import annotations
from fastai.basics import *
from fastai.callback.fp16 import AMPMode, MixedPrecision

from torch.cuda.amp import GradScaler

# +
#|default_exp callback.channelslast
# -

#|hide
from fastai.test_utils import *
from nbdev.showdoc import *


# # Channels Last training
# > Train models faster using channels last format (beta)

# With `MixedPrecision`, image models trained in channels last format on Tensor Cores can increase training throughput over contiguous format. PyTorch observed a [22% improvment](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains) in ResNet50 training speed using channels last and 8-35% improvement across a selection of models tested on a V100.
#
# Channels last format is compatible with modern GPUs (Volta, Turing, or newer) and modern CPUs (Ice Lake or newer).
#
# Channels last memory format currently is implemented for NCHW Tensors. Not all PyTorch operators have been converted to support channels last. See [(Beta) Channels Last Memory Format in PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) tutorial for more details.

# ## ChannelsLast -

#|export
class ChannelsLast(Callback):
    "Channels last training using PyTorch's Channels Last Memory Format (beta)"
    order = -1 # Needs to run before any model modification callbacks occur
    def before_fit(self):
        self.learn.model.to(memory_format=torch.channels_last)


# When a PyTorch model is set to channels last format, PyTorch will automatically convert any compatible NCHW input tensors to NHWC format. `ChannelsLast` sets the model to channels last format, so no changes to dataloaders or inputs are required.
#
# :::{.callout-note}
# `ChannelsLast` should work with most convolutional `timm` models.
#
# However, it is advised to test each model, as supported operations differ across PyTorch versions.
# :::
#
# Using `ChannelsLast` with unsupported PyTorch operations can lead to “channel thrashing”, where channels last input is converted to contiguous format in an unsupported PyTorch operation, then back to channels last for execution on the tensor core, back to contiguous when returned to the operation, and finally to channels last for the next layer. Too many unsupported operations in a model can lead to reduced performance.

#|export
#|export
@patch
@delegates(GradScaler)
def to_channelslast(self:Learner,
    use_amp:bool=True, # Add `MixedPrecision` with `amp_mode`. Recommended for full channels last performance
    amp_mode:str|AMPMode=AMPMode.FP16, # Mixed Precision training mode. Supports fp16 and bf16.
    **kwargs
):
    "Set `Learner` and inputs to `channels_last` format and float16 Mixed Precision by default"
    if use_amp and not hasattr(self, 'mixed_precision') and not hasattr(self, 'channels_last'):
        return self.add_cbs([ChannelsLast(), MixedPrecision(amp_mode, **kwargs)])
    elif not hasattr(self, 'channels_last'):
        return self.add_cb(ChannelsLast())


#|export
@patch
def to_contiguous(self:Learner, to_fp32:bool=False):
    "Set `Learner` and inputs to `contiguous_format` (default format), optionally to single precision"
    self.model.to(memory_format=torch.contiguous_format)
    if to_fp32:
        return self.remove_cbs([ChannelsLast, MixedPrecision])
    else:
        return self.remove_cb(ChannelsLast)


# ## Test Channels Last -

#|hide
from torch.utils.data import TensorDataset


#|hide
class ChannelsLastTest(Callback):
    "Asserts that predictions are in channels last format"
    order = MixedPrecision.order-1
    def after_pred(self):
        assert self.pred.is_contiguous(memory_format=torch.channels_last), "Model and/or output isn't channels last"


#|hide
#|cuda
def synth_dbunch(bs=16, n_train=10, n_valid=2, cuda=True):
    def get_data(n):
        return TensorDataset(TensorImage(torch.randn(bs*n, 3, 32, 32)))
    train_ds = get_data(n_train)
    valid_ds = get_data(n_valid)
    device = default_device() if cuda else None
    train_dl = TfmdDL(train_ds, bs=bs, shuffle=True, num_workers=0)
    valid_dl = TfmdDL(valid_ds, bs=bs, num_workers=0)
    return DataLoaders(train_dl, valid_dl, device=device)


#|hide
#|cuda
# Test must be ran on modern hardware (Volta, Turning, or newer)
with no_random():
    learn = synth_learner(cbs=[MixedPrecision,ChannelsLast,ChannelsLastTest], cuda=True, data=synth_dbunch())
    class ConvModel(Module):
        def __init__(self): self.conv = nn.Conv2d(3, 32, 1)
        def forward(self,x): return self.conv(x)
    def fakeloss(): pass
    learn.model = ConvModel()
    learn.opt_func = partial(SGD, mom=0.)
    learn.loss_func=fakeloss
    learn.fit(3)

# ## Export -

#|hide
from nbdev import *
nbdev_export()
