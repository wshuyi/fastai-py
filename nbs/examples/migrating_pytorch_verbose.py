# ---
# skip_exec: true
# jupyter:
#   jupytext:
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
#skip
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

# # Pytorch to fastai details
#
# > Step by step integrating raw PyTorch into the fastai framework

# In this tutorial we will be training MNIST (similar to the shortened tutorial [here](https://docs.fast.ai/migrating_pytorch.html)) from scratch using pure PyTorch and incrementally adding it to the fastai framework. What this entials is using:
# - PyTorch DataLoaders
# - PyTorch Model
# - PyTorch Optimizer
#
# And with fastai we will simply use the Training Loop (or the `Learner` class)
#
# In this tutorial also since generally people are more used to explicit exports, we will use explicit exports within the fastai library, but also do understand you can get all of these imports automatically by doing `from fastai.vision.all import *`
#
# > Generally it is also recommend you do so because of monkey-patching throughout the library, but this can be avoided as well which will be shown later.

# ## Data

# As mentioned in the title, we will  be loading in the dataset simply with the `torchvision` module. 
#
# This includes both loading in the dataset, and preparing it for the DataLoaders (including transforms)
#
# First we will grab our imports:

import torch, torchvision
import torchvision.transforms as transforms

# Next we can define some minimal transforms for converting the raw two-channel images into trainable tensors as well as normalize them:
#
# > The mean and standard deviation come from the MNIST dataset

tfms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081))
])

# Before finally creating our train and test `DataLoaders` by downloading the dataset and applying our transforms.

from torchvision import datasets
from torch.utils.data import DataLoader

# First let's download a train and test (or validation as it is reffered to in the fastai framework) dataset

train_dset = datasets.MNIST('../data', train=True, download=True, transform=tfms)
valid_dset = datasets.MNIST('../data', train=False, transform=tfms)

# Next we'll define a few hyperparameters to pass to the individual `DataLoader`'s as they are being made.
#
# We'll set a batch size of 256 while training, and 512 during the validation set
#
# We'll also use a single worker and pin the memory:

# +
train_loader = DataLoader(train_dset, batch_size=256, 
                          shuffle=True, num_workers=1, pin_memory=True)

test_loader = DataLoader(valid_dset, batch_size=512,
                         shuffle=False, num_workers=1, pin_memory=True)
# -

# Now we have raw PyTorch `DataLoader`'s. To use them within the fastai framework all that is left is to wrap it in the fastai `DataLoaders` class, which just takes in any number of `DataLoader` objects and combines them into one:

from fastai.data.core import DataLoaders

dls = DataLoaders(train_loader, test_loader)

# We have now prepared the data for `fastai`! Next let's build a basic model to use

# ## Model

# This will be an extremely simplistic 2 layer convolutional neural network with an extra set of layers that mimics fastai's generated `head`. In each head includes a `Flatten` layer, which simply just adjusts the shape of the outputs. We will mimic it here

from torch import nn


class Flatten(nn.Module):
    "Flattens an input"
    def forward(self, x): return x.view(x.size(0), -1)


# And then our actual model:

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), 
            # A head to the model
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            Flatten(), nn.Linear(9216, 128), nn.ReLU(),
            nn.Dropout2d(0.5), nn.Linear(128, 10), nn.LogSoftmax(dim=1)
        )


# ## Optimizer

# Using native PyTorch optimizers in the fastai framework is made extremely simple thanks to the `OptimWrapper` interface. 
#
# Simply write a `partial` function specifying the `opt` as a torch optimizer. 
#
# In our example we will use `Adam`:

# +
from fastai.optimizer import OptimWrapper

from torch import optim
from functools import partial
# -

opt_func = partial(OptimWrapper, opt=optim.Adam)

# And that is all that's needed to make a working optimizer in the framework. You do not need to declare layer groups or any of the sort, that all occurs in the `Learner` class which we will do next!

# ## Training

# Training in the fastai framework revolves around the `Learner` class. This class ties everything we declared earlier together and allows for quick training with many different schedulers and `Callback`'s quickly.  
# Basic way for import `Learner` is  
# `from fastai.learner import Learner`  
#
# Since we are using explicit exports in this tutorial, you will notice that we will import `Learner` different way. This is because `Learner` is heavily monkey-patched throughout the library, so to utilize it best we need to get all of the existing patches through importing the module.

import fastai.callback.schedule # To get `fit_one_cycle`, `lr_find`

# :::{.callout-note}
#
# All `Callbacks` will still work, regardless of the type of dataloaders. It is recommended to use the `.all` import when wanting so, this way all callbacks are imported and anything related to the `Learner` is imported at once as well
#
# :::

# To build the Learner (minimally), we need to pass in the `DataLoaders`, our model, a loss function, potentially some metrics to use, and an optimizer function. 
#
# Let's import the `accuracy` metric from fastai:

from fastai.metrics import accuracy

# We'll use `nll_loss` as our loss function as well

import torch.nn.functional as F

# And build our `Learner`:

learn = Learner(dls, Net(), loss_func=F.nll_loss, opt_func=opt_func, metrics=accuracy)

# Now that everything is tied together, let's train our model with the One-Cycle policy through the `fit_one_cycle` function. We'll also use a learning rate of 1e-2 for a single epoch
#
# It would be noted that fastai's training loop will automatically take care of moving tensors to the proper devices during training, and will use the GPU by default if it is available. When using non-fastai native individual DataLoaders, it will look at the model's device for what device we want to train with.

# To access any of the above parameters, we look in similarly-named properties such as `learn.dls`, `learn.model`, `learn.loss_func`, and so on. 

# Now let's train:

learn.fit_one_cycle(n_epoch=1, lr_max=1e-2)

# Now that we have trained our model, let's simulate shipping off the model to be used on inference or various prediction methods.

# ## Exporting and Predicting

# To export your trained model, you can either use the `learn.export` method coupled with `load_learner` to load it back in, but it should be noted that none of the inference API will work, as we did not train with the fastai data API.
#
# Instead you should save the model weights, and perform raw PyTorch inference.
#
# We will walk through a quick example below.
#
# First let's save the model weights:
#
# :::{.callout-note}
#
# Generally when doing this approach you should also store the source code to build the model as well
#
# :::

learn.save('myModel', with_opt=False)

# :::{.callout-note}
#
# `Learner.save` will save the optimizer state by default as well. When doing so the weights are located in the `model` key. We will set this to `false` for this tutorial
#
# :::

# You can see that it showed us the location where our trained weights were stored. Next, let's load that in as a seperated PyTorch model not tied to the `Learner`:

new_net = Net()
net_dict = torch.load('models/myModel.pth') 
new_net.load_state_dict(net_dict);

# Finally, let's predict on a single image using those `tfms` we declared earlier.
#
# When predicting in general we preprocess the dataset in the same form as the validation set, and this is how fastai does it as well with their `test_dl` and `test_set` methods.
#
# Since the downloaded dataset doesn't have individual files for us to work with, we will download a set of only 3's and 7's from fastai, and predict on one of those images:

from fastai.data.external import untar_data, URLs

data_path = untar_data(URLs.MNIST_SAMPLE)

data_path.ls()

# We'll grab one of the `valid` images

single_image = data_path/'valid'/'3'/'8483.png'

# Open it in Pillow:

from PIL import Image

im = Image.open(single_image)
im.load();

im

# Next we will apply the same transforms that we did to our validation set

tfmd_im = tfms(im); tfmd_im.shape

# We'll set it as a batch of 1:

tfmd_im = tfmd_im.unsqueeze(0)

tfmd_im.shape

# And then predict with our model:

with torch.no_grad():
    new_net.cuda()
    tfmd_im = tfmd_im.cuda()
    preds = new_net(tfmd_im)

# Let's look at the predictions:

preds

# This isn't quite what fastai outputs, we need to convert this into a class label to make it similar. To do so, we simply take the argmax of the predictions over the first index.
#
# If we were using fastai DataLoaders, it would use this as an index into a list of class names. Since our labels are 0-9, the argmax *is* our label:

preds.argmax(dim=-1)

# And we can see it correctly predicted a label of 3!
