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

# +
#|default_exp callback.mixup
# -

#|export
from __future__ import annotations
from fastai.basics import *
from torch.distributions.beta import Beta

#|hide
from nbdev.showdoc import *
from fastai.test_utils import *

# # MixUp and Friends
#
# > Callbacks that can apply the MixUp (and variants) data augmentation to your training

from fastai.vision.all import *


#|export
def reduce_loss(
    loss:Tensor, 
    reduction:str='mean' # PyTorch loss reduction
)->Tensor:
    "Reduce the loss based on `reduction`"
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


#|export
class MixHandler(Callback):
    "A handler class for implementing `MixUp` style scheduling"
    run_valid = False
    def __init__(self, 
        alpha:float=0.5 # Determine `Beta` distribution in range (0.,inf]
    ):
        self.distrib = Beta(tensor(alpha), tensor(alpha))

    def before_train(self):
        "Determine whether to stack y"
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y: self.old_lf,self.learn.loss_func = self.learn.loss_func,self.lf

    def after_train(self):
        "Set the loss function back to the previous loss"
        if self.stack_y: self.learn.loss_func = self.old_lf

    def after_cancel_train(self):
        "If training is canceled, still set the loss function back"
        self.after_train()

    def after_cancel_fit(self):
        "If fit is canceled, still set the loss function back"
        self.after_train()

    def lf(self, pred, *yb):
        "lf is a loss function that applies the original loss function on both outputs based on `self.lam`"
        if not self.training: return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))


# Most `Mix` variants will perform the data augmentation on the batch, so to implement your `Mix` you should adjust the `before_batch` event with however your training regiment requires. Also if a different loss function is needed, you should adjust the `lf` as well. `alpha` is passed to `Beta` to create a sampler.  

# ##  MixUp -

#|export
class MixUp(MixHandler):
    "Implementation of https://arxiv.org/abs/1710.09412"
    def __init__(self, 
        alpha:float=.4 # Determine `Beta` distribution in range (0.,inf]
    ): 
        super().__init__(alpha)
        
    def before_batch(self):
        "Blend xb and yb with another random item in a second batch (xb1,yb1) with `lam` weights"
        lam = self.distrib.sample((self.y.size(0),)).squeeze().to(self.x.device)
        lam = torch.stack([lam, 1-lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y.size(0)).to(self.x.device)
        xb1,self.yb1 = tuple(L(self.xb).itemgot(shuffle)),tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x.size())
        self.learn.xb = tuple(L(xb1,self.xb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=nx_dims-1)))

        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))


# This is a modified implementation of mixup that will always blend at least 50% of the original image.  The original paper calls for a Beta distribution which is passed the same value of alpha for each position in the loss function (alpha = beta = #).  Unlike the original paper, this implementation of mixup selects the max of lambda which means that if the value that is sampled as lambda is less than 0.5 (i.e the original image would be <50% represented, 1-lambda is used instead.  

# The blending of two images is determined by `alpha`.  
#
# $alpha=1.$:
#
# * All values between 0 and 1 have an equal chance of being sampled. 
# * Any amount of mixing between the two images is possible  
#
# $alpha<1.$:
#
# * The values closer to 0 and 1 become more likely to be sampled than the values near 0.5.  
# * It is more likely that one of the images will be selected with a slight amount of the other image.  
#
# $alpha>1.$:
#
# * The values closer to 0.5 become more likely than the numbers close to 0 or 1.
# * It is more likely that the images will be blended evenly.  

# First we'll look at a very minimalistic example to show how our data is being generated with the `PETS` dataset:

path = untar_data(URLs.PETS)
pat        = r'([^/]+)_\d+.*$'
fnames     = get_image_files(path/'images')
item_tfms  = [Resize(256, method='crop')]
batch_tfms = [*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
dls = ImageDataLoaders.from_name_re(path, fnames, pat, bs=64, item_tfms=item_tfms, 
                                    batch_tfms=batch_tfms)

# We can examine the results of our `Callback` by grabbing our data during `fit` at `before_batch` like so:

# +
mixup = MixUp(1.)
with Learner(dls, nn.Linear(3,4), loss_func=CrossEntropyLossFlat(), cbs=mixup) as learn:
    learn.epoch,learn.training = 0,True
    learn.dl = dls.train
    b = dls.one_batch()
    learn._split(b)
    learn('before_train')
    learn('before_batch')

_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(mixup.x,mixup.y), ctxs=axs.flatten())
# -

#|hide
test_ne(b[0], mixup.x)
test_eq(b[1], mixup.y)

# We can see that every so often an image gets "mixed" with another. 
#
# How do we train? You can pass the `Callback` either to `Learner` directly or to `cbs` in your fit function:

#|slow
learn = vision_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), metrics=[error_rate])
learn.fit_one_cycle(1, cbs=mixup)


# ## CutMix -

#|export
class CutMix(MixHandler):
    "Implementation of https://arxiv.org/abs/1905.04899"
    def __init__(self,
        alpha:float=1. # Determine `Beta` distribution in range (0.,inf]
    ):
        super().__init__(alpha)

    def before_batch(self):
        "Add `rand_bbox` patches with size based on `lam` and location chosen randomly."
        bs, _, H, W = self.x.size()
        self.lam = self.distrib.sample((1,)).to(self.x.device)
        shuffle = torch.randperm(bs).to(self.x.device)
        xb1,self.yb1 = self.x[shuffle], tuple((self.y[shuffle],))
        x1, y1, x2, y2 = self.rand_bbox(W, H, self.lam)
        self.learn.xb[0][..., y1:y2, x1:x2] = xb1[..., y1:y2, x1:x2]
        self.lam = (1 - ((x2-x1)*(y2-y1))/float(W*H))
        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))

    def rand_bbox(self,
        W:int, # Input image width
        H:int, # Input image height
        lam:Tensor # lambda sample from Beta distribution i.e tensor([0.3647])
    ) -> tuple: # Represents the top-left pixel location and the bottom-right pixel location
        "Give a bounding box location based on the size of the im and a weight"
        cut_rat = torch.sqrt(1. - lam).to(self.x.device)
        cut_w = torch.round(W * cut_rat).type(torch.long).to(self.x.device)
        cut_h = torch.round(H * cut_rat).type(torch.long).to(self.x.device)
        # uniform
        cx = torch.randint(0, W, (1,)).to(self.x.device)
        cy = torch.randint(0, H, (1,)).to(self.x.device)
        x1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='floor'), 0, W)
        y1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='floor'), 0, H)
        x2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='floor'), 0, W)
        y2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='floor'), 0, H)
        return x1, y1, x2, y2


# Similar to `MixUp`, `CutMix` will cut a random box out of two images and swap them together. We can look at a few examples below:

# +
cutmix = CutMix(1.)
with Learner(dls, nn.Linear(3,4), loss_func=CrossEntropyLossFlat(), cbs=cutmix) as learn:
    learn.epoch,learn.training = 0,True
    learn.dl = dls.train
    b = dls.one_batch()
    learn._split(b)
    learn('before_train')
    learn('before_batch')

_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(cutmix.x,cutmix.y), ctxs=axs.flatten())
# -

# We train with it in the exact same way as well

#|slow
learn = vision_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, error_rate])
learn.fit_one_cycle(1, cbs=cutmix)

# # Export - 

#|hide
from nbdev import nbdev_export
nbdev_export()


