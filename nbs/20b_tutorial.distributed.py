# ---
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
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

# # Notebook distributed training
# > Using `Accelerate` to launch a training script from your notebook

# + active=""
# ---
# skip_exec: true
# ---
# -

# ## Overview
#
# In this tutorial we will see how to use [Accelerate](https://github.com/huggingface/accelerate) to launch a training function on a distributed system, from inside your **notebook**! 
#
# To keep it easy, this example will follow training PETs, showcasing how all it takes is 3 new lines of code to be on your way!

# ## Setting up imports and building the DataLoaders
#
# First, make sure that Accelerate is installed on your system by running:
# ```bash
# pip install accelerate -U
# ```
#
# In your code, along with the normal `from fastai.module.all import *` imports two new ones need to be added:
# ```diff
# + from fastai.distributed import *
# from fastai.vision.all import *
# from fastai.vision.models.xresnet import *
#
# + from accelerate import notebook_launcher
# + from accelerate.utils import write_basic_config
# ```

# The first brings in the  `Learner.distrib_ctx` context manager. The second brings in Accelerate's [notebook_launcher](https://huggingface.co/docs/accelerate/launcher), the key function we will call to run what we want.

# +
#|hide
from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *

from accelerate import notebook_launcher
from accelerate.utils import write_basic_config
# -

# We need to setup `Accelerate` to use all of our GPUs. We can do so quickly with `write_basic_config ()`:
#
# :::{.callout-note}
#
# Since this checks `torch.cuda.device_count`, you will need to restart your notebook and skip calling this again to continue. It only needs to be ran once! Also if you choose not to use this run `accelerate config` from the terminal and set `mixed_precision` to `no`
#
# :::

# +
#from accelerate.utils import write_basic_config
#write_basic_config()
# -

# Next let's download some data to train on. You don't need to worry about using `rank0_first`, as since we're in our Jupyter Notebook it will only run on one process like normal:

path = untar_data(URLs.PETS)


# We wrap the creation of the `DataLoaders`, our `vision_learner`, and call to `fine_tune` inside of a `train` function. 
#
# :::{.callout-note}
#
# It is important to **not** build the `DataLoaders` outside of the function, as absolutely *nothing* can be loaded onto CUDA beforehand.
#
# :::

def get_y(o): return o[0].isupper()
def train(path):
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=get_y, item_tfms=Resize(224))
    learn = vision_learner(dls, resnet34, metrics=error_rate).to_fp16()
    learn.fine_tune(1)


# The last addition to the `train` function needed is to use our context manager before calling `fine_tune` and setting `in_notebook` to `True`:
#
# :::{.callout-note}
#
# for this example `sync_bn` is disabled for compatibility purposes with `torchvision`'s resnet34
#
# :::

def train(path):
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=get_y, item_tfms=Resize(224))
    learn = vision_learner(dls, resnet34, metrics=error_rate).to_fp16()
    with learn.distrib_ctx(sync_bn=False, in_notebook=True):
        learn.fine_tune(1)
    learn.export("pets")


# Finally, just call `notebook_launcher`, passing in the training function, any arguments as a tuple, and the number of GPUs (processes) to use:

notebook_launcher(train, (path,), num_processes=2)

# Afterwards we can import our exported `Learner`, save, or anything else we may want to do in our Jupyter Notebook outside of a distributed process

imgs = get_image_files(path)
learn = load_learner(path/'pets')
learn.predict(imgs[0])
