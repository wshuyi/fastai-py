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
from packaging.version import parse

from fastai.basics import *
from fastai.vision.core import *
from fastai.vision.data import *
from fastai.vision.augment import *
from fastai.vision import models

import torchvision
try: import timm
except ModuleNotFoundError: pass

# +
#|default_exp vision.learner
# -

#|hide
from nbdev.showdoc import *


# # Vision learner
#
# > All the functions necessary to build `Learner` suitable for transfer learning in computer vision

# The most important functions of this module are `vision_learner` and `unet_learner`. They will help you define a `Learner` using a pretrained model. See the [vision tutorial](23_tutorial.vision.ipynb) for examples of use.

# ## Cut a pretrained model

#|export
def _is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)


#|hide
m = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Linear(2,3), nn.Conv2d(2,3,1), nn.MaxPool3d(5))
test_eq([bool(_is_pool_type(m_)) for m_ in m.children()], [True,False,False,True])


# By default, the fastai library cuts a pretrained model at the pooling layer. This function helps detecting it. 

#|export
def has_pool_type(m):
    "Return `True` if `m` is a pooling layer or has one in its children"
    if _is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False


m = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Linear(2,3), nn.Conv2d(2,3,1), nn.MaxPool3d(5))
assert has_pool_type(m)
test_eq([has_pool_type(m_) for m_ in m.children()], [True,False,False,True])


#|export
def _get_first_layer(m):
    "Access first layer of a model"
    c,p,n = m,None,None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p,c=c,getattr(c,n)
    return c,p,n


#|export
def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    if n_in==1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
    elif n_in==2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:,:2] * 1.5
    else:
        # keep 3 channels weights and set others to null
        new_layer.weight.data[:,:3] = previous_layer.weight.data
        new_layer.weight.data[:,3:].zero_()


#|export
def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)


#|export
def cut_model(model, cut):
    "Cut an instantiated model"
    if   isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    raise NameError("cut must be either integer or a function")


#|export
def create_body(model, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    return cut_model(model, cut)


# `cut` can either be an integer, in which case we cut the model at the corresponding layer, or a function, in which case, this function returns `cut(model)`. It defaults to the first layer that contains some pooling otherwise.

# +
def tst(): return nn.Sequential(nn.Conv2d(3,5,3), nn.BatchNorm2d(5), nn.AvgPool2d(1), nn.Linear(3,4))
m = create_body(tst())
test_eq(len(m), 2)

m = create_body(tst(), cut=3)
test_eq(len(m), 3)

m = create_body(tst(), cut=noop)
test_eq(len(m), 4)

for n in range(1,5):    
    m = create_body(tst(), n_in=n)
    test_eq(_get_first_layer(m)[0].in_channels, n)


# -

# ## Head and model

#|export
def create_head(nf, n_out, lin_ftrs=None, ps=0.5, pool=True, concat_pool=True, first_bn=True, bn_final=False,
                lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    if pool and concat_pool: nf *= 2
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [True]*len(lin_ftrs[1:])
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = []
    if pool:
        pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
        layers += [pool, Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,bn,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


# The head begins with fastai's `AdaptiveConcatPool2d` if `concat_pool=True` otherwise, it uses traditional average pooling. Then it uses a `Flatten` layer before going on blocks of `BatchNorm`, `Dropout` and `Linear` layers (if `lin_first=True`, those are `Linear`, `BatchNorm`, `Dropout`).
#
# Those blocks start at `nf`, then every element of `lin_ftrs` (defaults to `[512]`) and end at `n_out`. `ps` is a list of probabilities used for the dropouts (if you only pass 1, it will use half the value then that value as many times as necessary).
#
# If `first_bn=True`, a `BatchNorm` added just after the pooling operations. If `bn_final=True`, a final `BatchNorm` layer is added. If `y_range` is passed, the function adds a `SigmoidRange` to that range.

tst = create_head(5, 10)
tst

# +
#|hide
mods = list(tst.children())
test_eq(len(mods), 9)
assert isinstance(mods[2], nn.BatchNorm1d)
assert isinstance(mods[-1], nn.Linear)

tst = create_head(5, 10, lin_first=True)
mods = list(tst.children())
test_eq(len(mods), 8)
assert isinstance(mods[2], nn.Dropout)

tst = create_head(5, 10, first_bn=False)
mods = list(tst.children())
test_eq(len(mods), 8)
assert isinstance(mods[2], nn.Dropout)

tst = create_head(5, 10, concat_pool=True)
modes = list(tst.children())
test_eq(modes[4].in_features, 10)

tst = create_head(5, 10, concat_pool=False)
modes = list(tst.children())
test_eq(modes[4].in_features, 5)
# -

#|export
from fastai.callback.hook import num_features_model


# +
#TODO: refactor, i.e. something like this?
# class ModelSplitter():
#     def __init__(self, idx): self.idx = idx
#     def split(self, m): return L(m[:self.idx], m[self.idx:]).map(params)
#     def __call__(self,): return {'cut':self.idx, 'split':self.split}
# -

#|export
def default_split(m):
    "Default split of a model between body and head"
    return L(m[0], m[1:]).map(params)


# To do transfer learning, you need to pass a `splitter` to `Learner`. This should be a function taking the model and returning a collection of parameter groups, e.g. a list of list of parameters.

# +
#|export
def _xresnet_split(m): return L(m[0][:3], m[0][3:], m[1:]).map(params)
def  _resnet_split(m): return L(m[0][:6], m[0][6:], m[1:]).map(params)
def _squeezenet_split(m:nn.Module): return L(m[0][0][:5], m[0][0][5:], m[1:]).map(params)
def _densenet_split(m:nn.Module): return L(m[0][0][:7],m[0][0][7:], m[1:]).map(params)
def _vgg_split(m:nn.Module): return L(m[0][0][:22], m[0][0][22:], m[1:]).map(params)
def _alexnet_split(m:nn.Module): return L(m[0][0][:6], m[0][0][6:], m[1:]).map(params)

_default_meta    = {'cut':None, 'split':default_split}
_xresnet_meta    = {'cut':-4, 'split':_xresnet_split, 'stats':imagenet_stats}
_resnet_meta     = {'cut':-2, 'split':_resnet_split, 'stats':imagenet_stats, 'weights':'DEFAULT'}
_squeezenet_meta = {'cut':-1, 'split': _squeezenet_split, 'stats':imagenet_stats, 'weights':'DEFAULT'}
_densenet_meta   = {'cut':-1, 'split':_densenet_split, 'stats':imagenet_stats, 'weights':'DEFAULT'}
_vgg_meta        = {'cut':-2, 'split':_vgg_split, 'stats':imagenet_stats, 'weights':'DEFAULT'}
_alexnet_meta    = {'cut':-2, 'split':_alexnet_split, 'stats':imagenet_stats, 'weights':'DEFAULT'}
# -

#|export
model_meta = {
    models.xresnet.xresnet18 :{**_xresnet_meta}, models.xresnet.xresnet34: {**_xresnet_meta},
    models.xresnet.xresnet50 :{**_xresnet_meta}, models.xresnet.xresnet101:{**_xresnet_meta},
    models.xresnet.xresnet152:{**_xresnet_meta},

    models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
    models.resnet152:{**_resnet_meta},

    models.squeezenet1_0:{**_squeezenet_meta},
    models.squeezenet1_1:{**_squeezenet_meta},

    models.densenet121:{**_densenet_meta}, models.densenet169:{**_densenet_meta},
    models.densenet201:{**_densenet_meta}, models.densenet161:{**_densenet_meta},
    models.vgg11_bn:{**_vgg_meta}, models.vgg13_bn:{**_vgg_meta}, models.vgg16_bn:{**_vgg_meta}, models.vgg19_bn:{**_vgg_meta},
    models.alexnet:{**_alexnet_meta}}


#|export
def add_head(body, nf, n_out, init=nn.init.kaiming_normal_, head=None, concat_pool=True, pool=True,
                lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None):
    "Add a head to a vision body"
    if head is None:
        head = create_head(nf, n_out, concat_pool=concat_pool, pool=pool,
                           lin_ftrs=lin_ftrs, ps=ps, first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range)
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model


#|export
def create_vision_model(arch, n_out, pretrained=True, weights=None, cut=None, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                        concat_pool=True, pool=True, lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None):
    "Create custom vision architecture"
    meta = model_meta.get(arch, _default_meta)
    if parse(torchvision.__version__) >= parse('0.13') and 'weights' in meta:
        if weights is not None and not pretrained:
            warn(f'{pretrained=} but `weights` are set {weights=}. To randomly initialize set `pretrained=False` & `weights=None`')
        model = arch(weights=meta['weights'] if (weights is None and pretrained) else weights)
    else:
        model = arch(pretrained=pretrained)
    body = create_body(model, n_in, pretrained, ifnone(cut, meta['cut']))
    nf = num_features_model(nn.Sequential(*body.children())) if custom_head is None else None
    return add_head(body, nf, n_out, init=init, head=custom_head, concat_pool=concat_pool, pool=pool,
                    lin_ftrs=lin_ftrs, ps=ps, first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range)


show_doc(create_vision_model)

# The model is cut according to `cut` and it may be `pretrained`, in which case, the proper set of weights is downloaded then loaded. `init` is applied to the head of the model, which is either created by `create_head` (with `lin_ftrs`, `ps`, `concat_pool`, `bn_final`, `lin_first` and `y_range`) or is `custom_head`.

tst = create_vision_model(models.resnet18, 10, True)
tst = create_vision_model(models.resnet18, 10, True, n_in=1)


#|export
class TimmBody(nn.Module):
    def __init__(self, model, pretrained:bool=True, cut=None, n_in:int=3):
        super().__init__()
        self.needs_pool = model.default_cfg.get('pool_size', None) is not None
        self.model = model if cut is None else cut_model(model, cut)
    
    def forward(self,x): return self.model.forward_features(x) if self.needs_pool else self.model(x)


#|export
def create_timm_model(arch, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, pool=True, lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    model = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=n_in, **kwargs)
    body = TimmBody(model, pretrained, None, n_in)
    nf = body.model.num_features
    res = add_head(body, nf, n_out, init=init, head=custom_head, concat_pool=concat_pool, pool=body.needs_pool,
                   lin_ftrs=lin_ftrs, ps=ps, first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range)
    return res,model.default_cfg


# make sure that timm models can be scripted:
tst, _ = create_timm_model('resnet34', 1)
scripted = torch.jit.script(tst)
assert scripted, "model could not be converted to TorchScript"


# ## `Learner` convenience functions

#|export
def _add_norm(dls, meta, pretrained, n_in=3):
    if not pretrained: return
    stats = meta.get('stats')
    if stats is None: return
    if n_in != len(stats[0]): return
    if not dls.after_batch.fs.filter(risinstance(Normalize)):
        dls.add_tfms([Normalize.from_stats(*stats)],'after_batch')


#|hide
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path/"images"), r'^(.*)_\d+.jpg$', item_tfms=Resize(224))
for _ in range(5): _add_norm(dls, model_meta[models.resnet34], True)
test_eq(len(dls.after_batch.fs), 2)


#|export
def _timm_norm(dls, cfg, pretrained, n_in=3):
    if not pretrained: return
    if n_in != len(cfg['mean']): return
    if not dls.after_batch.fs.filter(risinstance(Normalize)):
        tfm = Normalize.from_stats(cfg['mean'],cfg['std'])
        dls.add_tfms([tfm],'after_batch')


#|export
@delegates(create_vision_model)
def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, weights=None,
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
        model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
        # model & head args
        cut=None, init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, pool=True,
        lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Build a vision learner from `dls` and `arch`"
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    meta = model_meta.get(arch, _default_meta)
    model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                      first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    n_in = kwargs['n_in'] if 'n_in' in kwargs else 3
    if isinstance(arch, str):
        model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
        if normalize: _timm_norm(dls, cfg, pretrained, n_in)
    else:
        if normalize: _add_norm(dls, meta, pretrained, n_in)
        model = create_vision_model(arch, n_out, pretrained=pretrained, weights=weights, **model_args)

    splitter = ifnone(splitter, meta['split'])
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    return learn


# The model is built from `arch` using the number of final activations inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`).
#
# If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.
#
# All other arguments are passed to `Learner`.
#
# Starting with version 0.13, TorchVision supports [multiple pretrained weights](https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models) for the same model architecture. The <code>vision_learner</code> default of `pretrained=True, weights=None` will use the architecture's default weights, which are currently IMAGENET1K_V2. If you are using an older version of TorchVision or creating a [timm](https://huggingface.co/docs/timm/index) model, setting `weights` will have no effect.
#
# ```python
# from torchvision.models import ResNet50_Weights
#
# # Legacy weights with accuracy 76.130%
# vision_learner(models.resnet50, pretrained=True, weights=ResNet50_Weights.IMAGENET1K_V1, ...)
#
# # New weights with accuracy 80.858%. Strings are also supported.
# vision_learner(models.resnet50, pretrained=True, weights='IMAGENET1K_V2', ...)
#
# # Best available weights (currently an alias for IMAGENET1K_V2).
# # Default weights if vision_learner weights isn't set.
# vision_learner(models.resnet50, pretrained=True, weights=ResNet50_Weights.DEFAULT, ...)
#
# # No weights - random initialization
# vision_learner(models.resnet50, pretrained=False, weights=None, ...)
# ```
#
# The example above shows how to use the new TorchVision 0.13 multi-weight api with <code>vision_learner</code>.

path = untar_data(URLs.PETS)
fnames = get_image_files(path/"images")
pat = r'^(.*)_\d+.jpg$'
dls = ImageDataLoaders.from_name_re(path, fnames, pat, item_tfms=Resize(224))

learn = vision_learner(dls, models.resnet18, loss_func=CrossEntropyLossFlat(), ps=0.25)

#|hide
if parse(torchvision.__version__) >= parse('0.13'):
    from torchvision.models import ResNet34_Weights
    weights = ResNet34_Weights.IMAGENET1K_V1
else:
    weights = None

#|hide
learn = vision_learner(dls, models.resnet34, weights=weights, loss_func=CrossEntropyLossFlat(), ps=0.25, concat_pool=False)
test_ne(learn.cbs, None)

#|hide
test_eq(to_cpu(dls.after_batch[1].mean[0].squeeze()), tensor(imagenet_stats[0]))
test_eq(to_cpu(dls.valid.after_batch[1].mean[0].squeeze()), tensor(imagenet_stats[0]))

# If you pass a `str` to `arch`, then a [timm](https://huggingface.co/docs/timm/index) model will be created:

dls = ImageDataLoaders.from_name_re(path, fnames, pat, item_tfms=Resize(224))
learn = vision_learner(dls, 'convnext_tiny', loss_func=CrossEntropyLossFlat(), ps=0.25)


#|export
@delegates(models.unet.DynamicUnet.__init__)
def create_unet_model(arch, n_out, img_size, pretrained=True, weights=None, cut=None, n_in=3, **kwargs):
    "Create custom unet architecture"
    meta = model_meta.get(arch, _default_meta)
    if parse(torchvision.__version__) >= parse('0.13') and 'weights' in meta:
        if weights is not None and not pretrained:
            warn(f'{pretrained=} but `weights` are set {weights=}. To randomly initialize set `pretrained=False` & `weights=None`')
        model = arch(weights=meta['weights'] if (weights is None and pretrained) else weights)
    else:
        model = arch(pretrained=pretrained)
    body = create_body(model, n_in, pretrained, ifnone(cut, meta['cut']))
    model = models.unet.DynamicUnet(body, n_out, img_size, **kwargs)
    return model


show_doc(create_unet_model)

tst = create_unet_model(models.resnet18, 10, (24,24), True, n_in=1)


#|export
@delegates(create_unet_model)
def unet_learner(dls, arch, normalize=True, n_out=None, pretrained=True, weights=None, config=None,
                 # learner args
                 loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
                 model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95), **kwargs):
    "Build a unet learner from `dls` and `arch`"

    if config:
        warnings.warn('config param is deprecated. Pass your args directly to unet_learner.')
        kwargs = {**config, **kwargs}

    meta = model_meta.get(arch, _default_meta)
    n_in = kwargs['n_in'] if 'n_in' in kwargs else 3
    if normalize: _add_norm(dls, meta, pretrained, n_in)

    n_out = ifnone(n_out, get_c(dls))
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    img_size = dls.one_batch()[0].shape[-2:]
    assert img_size, "image size could not be inferred from data"
    model = create_unet_model(arch, n_out, img_size, pretrained=pretrained, weights=weights, **kwargs)

    splitter = ifnone(splitter, meta['split'])
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
                   moms=moms)
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    return learn


# The model is built from `arch` using the number of final filters inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`).
#
# If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.
#
# All other arguments are passed to `Learner`.
#
# <code>unet_learner</code> also supports TorchVision's new multi-weight API via `weights`. See `vision_learner` for more details.

path = untar_data(URLs.CAMVID_TINY)
fnames = get_image_files(path/'images')
def label_func(x): return path/'labels'/f'{x.stem}_P{x.suffix}'
codes = np.loadtxt(path/'codes.txt', dtype=str)
dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, codes=codes)

learn = unet_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(axis=1), y_range=(0,1))

#|hide
test_ne(learn.cbs, None)


#|export
def create_cnn_model(*args, **kwargs):
    "Deprecated name for `create_vision_model` -- do not use"
    warn("`create_cnn_model` has been renamed to `create_vision_model` -- please update your code")
    return create_vision_model(*args, **kwargs)


#|export
def cnn_learner(*args, **kwargs):
    "Deprecated name for `vision_learner` -- do not use"
    warn("`cnn_learner` has been renamed to `vision_learner` -- please update your code")
    return vision_learner(*args, **kwargs)


# ## Show functions -

#|export
@typedispatch
def show_results(x:TensorImage, y, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs


#|export
@typedispatch
def show_results(x:TensorImage, y:TensorCategory, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(2):
        ctxs = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs,range(max_n))]
    ctxs = [r.show(ctx=c, color='green' if b==r else 'red', **kwargs)
            for b,r,c,_ in zip(samples.itemgot(1),outs.itemgot(0),ctxs,range(max_n))]
    return ctxs


#|export
@typedispatch
def show_results(x:TensorImage, y:TensorMask|TensorPoint|TensorBBox, samples, outs, ctxs=None, max_n=6,
                 nrows=None, ncols=1, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize, double=True,
                                     title='Target/Prediction')
    for i in range(2):
        ctxs[::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs[::2],range(2*max_n))]
    for o in [samples,outs]:
        ctxs[1::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(o.itemgot(0),ctxs[1::2],range(2*max_n))]
    return ctxs


#|export
@typedispatch
def show_results(x:TensorImage, y:TensorImage, samples, outs, ctxs=None, max_n=10, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(3*min(len(samples), max_n), ncols=3, figsize=figsize, title='Input/Target/Prediction')
    for i in range(2):
        ctxs[i::3] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(i),ctxs[i::3],range(max_n))]
    ctxs[2::3] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(outs.itemgot(0),ctxs[2::3],range(max_n))]
    return ctxs


#|export
@typedispatch
def plot_top_losses(x: TensorImage, y:TensorCategory, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, figsize=figsize, title='Prediction/Actual/Loss/Probability')
    for ax,s,o,r,l in zip(axs, samples, outs, raws, losses):
        s[0].show(ctx=ax, **kwargs)
        ax.set_title(f'{o[0]}/{s[1]} / {l.item():.2f} / {r.max().item():.2f}')


#|export
@typedispatch
def plot_top_losses(x: TensorImage, y:TensorMultiCategory, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, figsize=figsize)
    for i,(ax,s) in enumerate(zip(axs, samples)): s[0].show(ctx=ax, title=f'Image {i}', **kwargs)
    rows = get_empty_df(len(samples))
    outs = L(s[1:] + o + (TitledStr(r), TitledFloat(l.item())) for s,o,r,l in zip(samples, outs, raws, losses))
    for i,l in enumerate(["target", "predicted", "probabilities", "loss"]):
        rows = [b.show(ctx=r, label=l, **kwargs) for b,r in zip(outs.itemgot(i),rows)]
    display_df(pd.DataFrame(rows))


#|export
@typedispatch
def plot_top_losses(x:TensorImage, y:TensorMask, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axes = get_grid(len(samples)*3, nrows=len(samples), ncols=3, figsize=figsize, flatten=False, title="Input | Target | Prediction")
    if axes.ndim == 1: axes = (axes,)
    titles = ["input", "target", "pred"]
    for axs,s,o,l in zip(axes, samples, outs, losses):
        imgs = (s[0], s[1], o[0])
        for ax,im,title in zip(axs, imgs, titles):
            if title=="pred": title += f"; loss = {l.item():.4f}"
            im.show(ctx=ax, **kwargs)
            ax.set_title(title)


# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()
