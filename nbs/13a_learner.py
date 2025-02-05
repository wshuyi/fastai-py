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
#|default_exp learner
# -

#|export
from __future__ import annotations
from fastai.data.all import *
from fastai.optimizer import *
from fastai.callback.core import *
import pickle,threading
from collections.abc import MutableSequence

#|hide
from nbdev.showdoc import *

#|export
_all_ = ['CancelBackwardException', 'CancelStepException','CancelFitException','CancelEpochException',
         'CancelTrainException','CancelValidException','CancelBatchException']

# # Learner, Metrics, Callbacks
#
# > Basic class for handling the training loop

# You probably want to jump directly to the definition of `Learner`.

# ## Utils function

#|hide
#For tests
from torch.utils.data import TensorDataset, DataLoader as TorchDL


# +
#|hide
def synth_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2, cuda=False, tfmdDL=True):
    "A simple dataset where `x` is random and `y = a*x + b` plus some noise."
    def get_data(n):
        x = torch.randn(int(bs*n))
        return TensorDataset(x, a*x + b + 0.1*torch.randn(int(bs*n)))
    train_ds = get_data(n_train)
    valid_ds = get_data(n_valid)
    device = default_device() if cuda else None
    if tfmdDL:
        train_dl = TfmdDL(train_ds, bs=bs, shuffle=True, num_workers=0, drop_last=True)
        valid_dl = TfmdDL(valid_ds, bs=bs, num_workers=0)
    else:
        train_dl = TorchDL(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        valid_dl = TorchDL(valid_ds, batch_size=bs, num_workers=0)
        device = None
    return DataLoaders(train_dl, valid_dl, device=device)

class RegModel(Module):
    "A r"
    def __init__(self): self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
    def forward(self, x): return x*self.a + self.b


# -

#|export
defaults.lr = 1e-3


#|export
def replacing_yield(o, attr, val):
    "Context manager to temporarily replace an attribute"
    old = getattr(o,attr)
    try:     yield setattr(o,attr,val)
    finally: setattr(o,attr,old)


# +
class _A:
    def __init__(self, a): self.a = a
    @contextmanager
    def a_changed(self, v): return replacing_yield(self, 'a', v)

a = _A(42)
with a.a_changed(32):
    test_eq(a.a, 32)
test_eq(a.a, 42)


# -

#|export
def mk_metric(m):
    "Convert `m` to an `AvgMetric`, unless it's already a `Metric`"
    if isinstance(m,type): m = m()
    return m if isinstance(m, Metric) else AvgMetric(m)


# See the class `Metric` below for more information.

#|export
def save_model(file, model, opt, with_opt=True, pickle_protocol=2, **torch_save_kwargs):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if rank_distrib(): return # don't save if child proc
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, file, pickle_protocol=pickle_protocol, **torch_save_kwargs)


# `file` can be a `Path` object, a string or an opened file object. `pickle_protocol` and `torch_save_kwargs` is passed along to `torch.save`

#|export
def load_model(file, model, opt, with_opt=True, device=None, strict=True, **torch_load_kwargs):
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    if isinstance(device, int): device = torch.device('cuda', device)
    elif device is None: device = 'cpu'
    state = torch.load(file, map_location=device, **torch_load_kwargs)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if hasopt and with_opt:
        try: opt.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved file doesn't contain an optimizer state.")


# `file` can be a `Path` object, a string or an opened file object. If a `device` is passed, the model is loaded on it, otherwise it's loaded on the CPU. 
#
# If `strict` is `True`, the file must exactly contain weights for every parameter key in `model`, if `strict` is `False`, only the keys that are in the saved model are loaded in `model`.
#
# You can pass in other kwargs to `torch.load` through `torch_load_kwargs`.

#|export
def _try_concat(o):
    try:    return torch.cat(o)
    except: return sum([L(o_[i,:] for i in range_of(o_)) for o_ in o], L())


#|export
_before_epoch = [event.before_fit, event.before_epoch]
_after_epoch  = [event.after_epoch, event.after_fit]


#|export
class _ConstantFunc():
    "Returns a function that returns `o`"
    def __init__(self, o): self.o = o
    def __call__(self, *args, **kwargs): return self.o


#|export
class SkipToEpoch(Callback):
    "Skip training up to `epoch`"
    order = 70
    
    def __init__(self, epoch:int):
        self._skip_to = epoch

    def before_epoch(self):
        if self.epoch < self._skip_to:
            raise CancelEpochException


# ## Learner -

#|export
_loop = ['Start Fit', 'before_fit', 'Start Epoch Loop', 'before_epoch', 'Start Train', 'before_train',
         'Start Batch Loop', 'before_batch', 'after_pred', 'after_loss', 'before_backward', 'before_step',
         'after_step', 'after_cancel_batch', 'after_batch','End Batch Loop','End Train',
         'after_cancel_train', 'after_train', 'Start Valid', 'before_validate','Start Batch Loop',
         '**CBs same as train batch**', 'End Batch Loop', 'End Valid', 'after_cancel_validate',
         'after_validate', 'End Epoch Loop', 'after_cancel_epoch', 'after_epoch', 'End Fit',
         'after_cancel_fit', 'after_fit']


# +
#|export
class Learner(GetAttr):
    _default='model'
    def __init__(self,
        dls:DataLoaders, # `DataLoaders` containing fastai or PyTorch `DataLoader`s
        model:callable, # PyTorch model for training or inference
        loss_func:callable|None=None, # Loss function. Defaults to `dls` loss
        opt_func:Optimizer|OptimWrapper=Adam, # Optimization function for training
        lr:float|slice=defaults.lr, # Default learning rate
        splitter:callable=trainable_params, # Split model into parameter groups. Defaults to one parameter group
        cbs:Callback|MutableSequence|None=None, # `Callback`s to add to `Learner`
        metrics:callable|MutableSequence|None=None, # `Metric`s to calculate on validation set
        path:str|Path|None=None, # Parent directory to save, load, and export models. Defaults to `dls` `path`
        model_dir:str|Path='models', # Subdirectory to save and load models
        wd:float|int|None=None, # Default weight decay
        wd_bn_bias:bool=False, # Apply weight decay to normalization and bias parameters
        train_bn:bool=True, # Train frozen normalization layers
        moms:tuple=(0.95,0.85,0.95), # Default momentum for schedulers
        default_cbs:bool=True # Include default `Callback`s
    ):
        path = Path(path) if path is not None else getattr(dls, 'path', Path('.'))
        if loss_func is None:
            loss_func = getattr(dls.train_ds, 'loss_func', None)
            assert loss_func is not None, "Could not infer loss function from the data, please pass a loss function."
        self.dls,self.model = dls,model
        store_attr(but='dls,model,cbs')
        self.training,self.create_mbar,self.logger,self.opt,self.cbs = False,True,print,None,L()
        if default_cbs: self.add_cbs(L(defaults.callbacks))
        self.add_cbs(cbs)
        self.lock = threading.Lock()
        self("after_create")

    @property
    def metrics(self): return self._metrics
    @metrics.setter
    def metrics(self,v): self._metrics = L(v).map(mk_metric)

    def _grab_cbs(self, cb_cls): return L(cb for cb in self.cbs if isinstance(cb, cb_cls))

    def add_cbs(self, cbs):
        L(cbs).map(self.add_cb)
        return self

    def remove_cbs(self, cbs):
        L(cbs).map(self.remove_cb)
        return self

    def add_cb(self, cb):
        if isinstance(cb, type): cb = cb()
        cb.learn = self
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return self

    def remove_cb(self, cb):
        if isinstance(cb, type): self.remove_cbs(self._grab_cbs(cb))
        else:
            cb.learn = None
            if hasattr(self, cb.name): delattr(self, cb.name)
            if cb in self.cbs: self.cbs.remove(cb)
        return self

    @contextmanager
    def added_cbs(self, cbs):
        self.add_cbs(cbs)
        try: yield
        finally: self.remove_cbs(cbs)

    @contextmanager
    def removed_cbs(self, cbs):
        self.remove_cbs(cbs)
        try: yield self
        finally: self.add_cbs(cbs)

    def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
    def __call__(self, event_name): L(event_name).map(self._call_one)

    def _call_one(self, event_name):
        if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
        for cb in self.cbs.sorted('order'): cb(event_name)

    def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

    def create_opt(self):
        if isinstance(self.opt_func, partial):
            if 'lr' in self.opt_func.keywords:
                self.lr = self.opt_func.keywords['lr']
        if isinstance(self.opt_func, OptimWrapper):
            self.opt = self.opt_func
            self.opt.clear_state()
        else:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True

    def _split(self, b):
        i = getattr(self.dls, 'n_inp', 1 if len(b)==1 else len(b)-1)
        self.xb,self.yb = b[:i],b[i:]

    def _with_events(self, f, event_type, ex, final=noop):
        try: self(f'before_{event_type}');  f()
        except ex: self(f'after_cancel_{event_type}')
        self(f'after_{event_type}');  final()

    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _backward(self): self.loss_grad.backward()
    def _step(self): self.opt.step()

    def _do_grad_opt(self):
        self._with_events(self._backward, 'backward', CancelBackwardException)
        self._with_events(self._step, 'step', CancelStepException)
        self.opt.zero_grad()

    def _do_one_batch(self):
        self.pred = self.model(*self.xb)
        self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return
        self._do_grad_opt()

    def _set_device(self, b):
        model_device = next(self.model.parameters()).device
        dls_device = getattr(self.dls, 'device', default_device())
        if model_device == dls_device: return to_device(b, dls_device)
        else: return to_device(b, model_device)

    def one_batch(self, i, b):
        self.iter = i
        b = self._set_device(b)
        self._split(b)
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)

    def _do_epoch_train(self):
        self.dl = self.dls.train
        self._with_events(self.all_batches, 'train', CancelTrainException)

    def _do_epoch_validate(self, ds_idx=1, dl=None):
        if dl is None: dl = self.dls[ds_idx]
        self.dl = dl
        with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)

    def _do_epoch(self):
        self._do_epoch_train()
        self._do_epoch_validate()

    def _do_fit(self):
        for epoch in range(self.n_epoch):
            self.epoch=epoch
            self._with_events(self._do_epoch, 'epoch', CancelEpochException)

    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):
        if start_epoch != 0:
            cbs = L(cbs) + SkipToEpoch(start_epoch)
        with self.added_cbs(cbs):
            if reset_opt or not self.opt: self.create_opt()
            if wd is None: wd = self.wd
            if wd is not None: self.opt.set_hypers(wd=wd)
            self.opt.set_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch
            self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

    def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None
    def __enter__(self): self(_before_epoch); return self
    def __exit__(self, exc_type, exc_value, tb): self(_after_epoch)

    def validation_context(self, cbs=None, inner=False):
        cms = [self.no_logging(),self.no_mbar(), self.lock]
        if cbs: cms.append(self.added_cbs(cbs))
        if not inner: cms.append(self)
        return ContextManagers(cms)

    def validate(self, ds_idx=1, dl=None, cbs=None):
        if dl is None: dl = self.dls[ds_idx]
        with self.validation_context(cbs=cbs): self._do_epoch_validate(ds_idx, dl)
        return getattr(self, 'final_record', None)

    @delegates(GatherPredsCallback.__init__)
    def get_preds(self,
        ds_idx:int=1, # `DataLoader` to use for predictions if `dl` is None. 0: train. 1: valid
        dl=None, # `DataLoader` to use for predictions, defaults to `ds_idx=1` if None
        with_input:bool=False, # Return inputs with predictions
        with_decoded:bool=False, # Return decoded predictions
        with_loss:bool=False, # Return per item loss with predictions
        act=None, # Apply activation to predictions, defaults to `self.loss_func`'s activation
        inner:bool=False, # If False, create progress bar, show logger, use temporary `cbs`
        reorder:bool=True, # Reorder predictions on dataset indicies, if applicable
        cbs:Callback|MutableSequence|None=None, # Temporary `Callback`s to apply during prediction
        **kwargs
    )-> tuple:
        if dl is None: dl = self.dls[ds_idx].new(shuffle=False, drop_last=False)
        else:
            try: len(dl)
            except TypeError as e:
                raise TypeError(f"`dl` is {type(dl)} and doesn't have len(dl)")
        if isinstance(dl, DataLoader):
            if dl.drop_last: dl = dl.new(shuffle=False, drop_last=False)
        if reorder and hasattr(dl, 'get_idxs'):
            idxs = dl.get_idxs()
            dl = dl.new(get_idxs = _ConstantFunc(idxs))
        cb = GatherPredsCallback(with_input=with_input, with_loss=with_loss, **kwargs)
        ctx_mgrs = self.validation_context(cbs=L(cbs)+[cb], inner=inner)
        if with_loss: ctx_mgrs.append(self.loss_not_reduced())
        with ContextManagers(ctx_mgrs):
            self._do_epoch_validate(dl=dl)
            if act is None: act = getcallable(self.loss_func, 'activation')
            res = cb.all_tensors()
            pred_i = 1 if with_input else 0
            if res[pred_i] is not None:
                res[pred_i] = act(res[pred_i])
                if with_decoded: res.insert(pred_i+2, getcallable(self.loss_func, 'decodes')(res[pred_i]))
            if reorder and hasattr(dl, 'get_idxs'): res = nested_reorder(res, tensor(idxs).argsort())
            return tuple(res)
        self._end_cleanup()

    def predict(self, item, rm_type_tfms=None, with_input=False):
        dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        i = getattr(self.dls, 'n_inp', -1)
        inp = (inp,) if i==1 else tuplify(inp)
        dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
        dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
        res = dec_targ,dec_preds[0],preds[0]
        if with_input: res = (dec_inp,) + res
        return res

    def show_results(self, ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
        if dl is None: dl = self.dls[ds_idx].new(shuffle=shuffle)
        b = dl.one_batch()
        _,_,preds = self.get_preds(dl=[b], with_decoded=True)
        dl.show_results(b, preds, max_n=max_n, **kwargs)

    def show_training_loop(self):
        indent = 0
        for s in _loop:
            if s.startswith('Start'): print(f'{" "*indent}{s}'); indent += 2
            elif s.startswith('End'): indent -= 2; print(f'{" "*indent}{s}')
            else: print(f'{" "*indent} - {s:15}:', self.ordered_cbs(s))

    @contextmanager
    def no_logging(self): return replacing_yield(self, 'logger', noop)
    @contextmanager
    def no_mbar(self):    return replacing_yield(self, 'create_mbar', False)

    @contextmanager
    def loss_not_reduced(self):
        if hasattr(self.loss_func, 'reduction'): return replacing_yield(self.loss_func, 'reduction', 'none')
        else: return replacing_yield(self, 'loss_func', partial(self.loss_func, reduction='none'))
    
    def to_detach(self,b,cpu=True,gather=True):
        return self.dl.to_detach(b,cpu,gather) if hasattr(getattr(self,'dl',None),'to_detach') else to_detach(b,cpu,gather)
    
    def __getstate__(self): return {k:v for k,v in self.__dict__.items() if k!='lock'}
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

Learner.x,Learner.y = add_props(lambda i,x: detuplify((x.xb,x.yb)[i]))
# -

#|export
add_docs(Learner, "Group together a `model`, some `dls` and a `loss_func` to handle training",
    add_cbs="Add `cbs` to the list of `Callback` and register `self` as their learner",
    add_cb="Add `cb` to the list of `Callback` and register `self` as their learner",
    remove_cbs="Remove `cbs` from the list of `Callback` and deregister `self` as their learner",
    remove_cb="Add `cb` from the list of `Callback` and deregister `self` as their learner",
    added_cbs="Context manage that temporarily adds `cbs`",
    removed_cbs="Context manage that temporarily removes `cbs`",
    ordered_cbs="List of `Callback`s, in order, for an `event` in the training loop",
    create_opt="Create an optimizer with default hyper-parameters",
    one_batch="Train or evaluate `self.model` on batch `(xb,yb)`",
    all_batches="Train or evaluate `self.model` on all the batches of `self.dl`",
    fit="Fit `self.model` for `n_epoch` using `cbs`. Optionally `reset_opt`.",
    validate="Validate on `dl` with potential new `cbs`.",
    get_preds="Get the predictions and targets on the `ds_idx`-th dbunchset or `dl`, optionally `with_input` and `with_loss`",
    predict="Prediction on `item`, fully decoded, loss function decoded and probabilities",
    validation_context="A `ContextManagers` suitable for validation, with optional `cbs`",
    show_results="Show some predictions on `ds_idx`-th dataset or `dl`",
    show_training_loop="Show each step in the training loop",
    no_logging="Context manager to temporarily remove `logger`",
    no_mbar="Context manager to temporarily prevent the master progress bar from being created",
    loss_not_reduced="A context manager to evaluate `loss_func` with reduction set to none.",
    to_detach="Calls `to_detach` if `self.dl` provides a `.to_detach` function otherwise calls global `to_detach`",
    __call__="Call `event_name` for all `Callback`s in `self.cbs`"
)

show_doc(Learner)

# `opt_func` will be used to create an optimizer when `Learner.fit` is called, with `lr` as a default learning rate. `splitter` is a function that takes `self.model` and returns a list of parameter groups (or just one parameter group if there are no different parameter groups). The default is `trainable_params`, which returns all trainable parameters of the model.
#
# `cbs` is one or a list of `Callback`s  to pass to the `Learner`. `Callback`s are used for every tweak of the training loop. Each `Callback` is registered as an attribute of `Learner` (with camel case). At creation, all the callbacks in `defaults.callbacks` (`TrainEvalCallback`, `Recorder` and `ProgressCallback`) are associated to the `Learner`.
#
# `metrics` is an optional list of metrics, that can be either functions or `Metric`s (see below). 
#
# `path` and `model_dir` are used to save and/or load models. Often `path` will be inferred from `dls`, but you can override it or pass a `Path`  object to `model_dir`. Make sure you can write in `path/model_dir`!
#
# `wd` is the default weight decay used when training the model; `moms`, the default momentums used in `Learner.fit_one_cycle`. `wd_bn_bias` controls if weight decay is applied to `BatchNorm` layers and bias. 
#
# Lastly, `train_bn` controls if `BatchNorm` layers are trained even when they are supposed to be frozen according to the `splitter`. Our empirical experiments have shown that it's the best behavior for those layers in transfer learning.

# ### PyTorch interop

# You can use regular PyTorch functionality for most of the arguments of the `Learner`, although the experience will be smoother with pure fastai objects and you will be able to use the full functionality of the library. The expectation is that the training loop will work smoothly even if you did not use fastai end to end. What you might lose are interpretation objects or showing functionality. The list below explains how to use plain PyTorch objects for all the arguments and what you might lose.
#
# The most important is `opt_func`. If you are not using a fastai optimizer, you will need to write a function that wraps your PyTorch optimizer in an `OptimWrapper`. See the [optimizer module](http://docs.fast.ai/optimizer.html) for more details. This is to ensure the library's schedulers/freeze API work with your code.
#
# - `dls` is a `DataLoaders` object, that you can create from standard PyTorch dataloaders. By doing so, you will lose all showing functionality like `show_batch`/`show_results`. You can check the [data block API](http://docs.fast.ai/tutorial.datablock.html) or the [mid-level data API tutorial](http://docs.fast.ai/tutorial.pets.html) to learn how to use fastai to gather your data!
# - `model` is a standard PyTorch model. You can use anyone you like, just make sure it accepts the number of inputs you have in your `DataLoaders` and returns as many outputs as you have targets.
# - `loss_func` can be any loss function you like. It needs to be one of fastai's if you want to use `Learn.predict` or `Learn.get_preds`, or you will have to implement special methods (see more details after the `BaseLoss` documentation).

# ### Training loop

# Now let's look at the main thing the `Learner` class implements: the training loop.

#|export
if not hasattr(defaults, 'callbacks'): defaults.callbacks = [TrainEvalCallback]

show_doc(Learner.fit)


# Uses `lr` and `wd` if they are provided, otherwise use the defaults values given by the `lr` and `wd` attributes of `Learner`.

# All the examples use `synth_learner` which is a simple `Learner` training a linear regression model.

#|hide
def synth_learner(n_train=10, n_valid=2, cuda=False, tfmdDL=True, lr=defaults.lr, **kwargs):
    data = synth_dbunch(n_train=n_train,n_valid=n_valid, cuda=cuda, tfmdDL=tfmdDL)
    return Learner(data, RegModel(), loss_func=MSELossFlat(), lr=lr, **kwargs)


#Training a few epochs should make the model better
learn = synth_learner(lr=0.1)
learn(_before_epoch)
learn.model = learn.model.cpu()
xb,yb = learn.dls.one_batch()
init_loss = learn.loss_func(learn.model(xb), yb)
learn.fit(10)
xb,yb = learn.dls.one_batch()
final_loss = learn.loss_func(learn.model(xb), yb)
assert final_loss < init_loss, (final_loss,init_loss)

#|hide
#Ensure we can train with raw torch
learn = synth_learner(lr=0.1, tfmdDL=False)
learn(_before_epoch)
learn.model = learn.model.cpu()
xb,yb = next(iter(learn.dls[0]))
init_loss = learn.loss_func(learn.model(xb), yb)
learn.fit(10)
xb,yb = next(iter(learn.dls[0]))
learn.model = learn.model.cpu() # Ensure we're still on CPU even in a CUDA environment
final_loss = learn.loss_func(learn.model(xb), yb)
assert final_loss < init_loss, (final_loss,init_loss)


# +
#|hide
class TestTrainEvalCallback(Callback):
    run_after,run_valid = TrainEvalCallback,False
    def before_fit(self): 
        test_eq([self.pct_train,self.train_iter], [0., 0])
        self.old_pct_train,self.old_train_iter = self.pct_train,self.train_iter
    
    def before_batch(self): test_eq(next(self.parameters()).device, find_device(self.xb))
    
    def after_batch(self):
        assert self.training
        test_eq(self.pct_train , self.old_pct_train+1/(self.n_iter*self.n_epoch))
        test_eq(self.train_iter, self.old_train_iter+1)
        self.old_pct_train,self.old_train_iter = self.pct_train,self.train_iter
    
    def before_train(self):
        assert self.training and self.model.training
        test_eq(self.pct_train, self.epoch/self.n_epoch)
        self.old_pct_train = self.pct_train
    
    def before_validate(self):
        assert not self.training and not self.model.training
        
learn = synth_learner(cbs=TestTrainEvalCallback)
learn.fit(1)
#Check order is properly taken into account
learn.cbs = L(reversed(learn.cbs))
# -

#|hide
#|cuda
#Check model is put on the GPU if needed
learn = synth_learner(cbs=TestTrainEvalCallback, cuda=True)
learn.fit(1)

#|hide
#|cuda
#Check that raw DataLoaders are put on the GPU
learn = synth_learner(cbs=TestTrainEvalCallback, tfmdDL=False)
learn.fit(1)


# +
#|hide
#Check wd is not applied on bn/bias when option wd_bn_bias=False
class _TstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
        self.tst = nn.Sequential(nn.Linear(4,5), nn.BatchNorm1d(3))
        self.tst[0].bias.data,self.tst[1].bias.data = torch.randn(5),torch.randn(3) 
    def forward(self, x): return x * self.a + self.b
    
class _PutGrad(Callback):
    def before_step(self):
        for p in self.learn.tst.parameters():
            p.grad = torch.ones_like(p.data)
    
learn = synth_learner(n_train=5, opt_func = partial(SGD, wd=1, decouple_wd=True), cbs=_PutGrad)
learn.model = _TstModel()
init = [p.clone() for p in learn.tst.parameters()]
learn.fit(1, lr=1e-2)
end = list(learn.tst.parameters())
assert not torch.allclose(end[0]-init[0], -0.05 * torch.ones_like(end[0]))
for i in [1,2,3]: test_close(end[i]-init[i], -0.05 * torch.ones_like(end[i]))
# -

show_doc(Learner.one_batch)

# This is an internal method called by `Learner.fit`. If passed, `i` is the index of this iteration in the epoch. In training mode, this does a full training step on the batch (compute predictions, loss, gradients, update the model parameters and zero the gradients). In validation mode, it stops at the loss computation. Training or validation is controlled internally by the `TrainEvalCallback` through the `training` attribute.
#
# Nothing is returned, but the attributes `x`, `y`, `pred`, `loss` of the `Learner` are set with the proper values:

b = learn.dls.one_batch()
learn.one_batch(0, b)
test_eq(learn.x, b[0])
test_eq(learn.y, b[1])
out = learn.model(learn.x)
test_eq(learn.pred, out)
test_eq(learn.loss, learn.loss_func(out, b[1]))


#|hide
class VerboseCallback(Callback):
    "Callback that prints the name of each event called"
    def __call__(self, event_name):
        print(event_name)
        super().__call__(event_name)


#|hide
class TestOneBatch(VerboseCallback):
    def __init__(self, xb, yb, i):
        self.save_xb,self.save_yb,self.i = xb,yb,i
        self.old_pred,self.old_loss = None,tensor(0.)
        
    def before_batch(self):
        self.old_a,self.old_b = self.a.data.clone(),self.b.data.clone()
        test_eq(self.iter,    self.i)
        test_eq(self.save_xb, *self.xb)
        test_eq(self.save_yb, *self.yb)
        if hasattr(self.learn, 'pred'): test_eq(self.pred, self.old_pred)
    
    def after_pred(self):
        self.old_pred = self.pred
        test_eq(self.pred, self.a.data * self.x + self.b.data)
        test_eq(self.loss, self.old_loss)
    
    def after_loss(self):
        self.old_loss = self.loss
        test_eq(self.loss, self.loss_func(self.old_pred, self.save_yb))
        for p in self.parameters(): 
            if not hasattr(p, 'grad') or p.grad is not None: test_eq(p.grad, tensor([0.]))
    
    def before_step(self):
        self.grad_a = (2 * self.x * (self.pred.data - self.y)).mean()
        self.grad_b = 2 * (self.pred.data - self.y).mean()
        test_close(self.a.grad.data, self.grad_a)
        test_close(self.b.grad.data, self.grad_b)
        test_eq(self.a.data, self.old_a)
        test_eq(self.b.data, self.old_b)
        
    def after_step(self):
        test_close(self.a.data, self.old_a - self.lr * self.grad_a)
        test_close(self.b.data, self.old_b - self.lr * self.grad_b)
        self.old_a,self.old_b = self.a.data.clone(),self.b.data.clone()
        test_close(self.a.grad.data, self.grad_a)
        test_close(self.b.grad.data, self.grad_b)
    
    def after_batch(self):
        for p in self.parameters(): test_eq(p.grad, tensor([0.]))


#|hide
learn = synth_learner()
b = learn.dls.one_batch()
learn = synth_learner(cbs=TestOneBatch(*b, 42), lr=1e-2)
#Remove train/eval
learn.cbs = learn.cbs[1:]
#Setup
learn.loss,learn.training = tensor(0.),True
learn.opt = SGD(learn.parameters(), lr=learn.lr)
learn.model.train()
batch_events = ['before_batch', 'after_pred', 'after_loss', 'before_backward', 'after_backward', 'before_step', 'after_step', 'after_batch']
test_stdout(lambda: learn.one_batch(42, b), '\n'.join(batch_events))
test_stdout(lambda: learn.one_batch(42, b), '\n'.join(batch_events)) #Check it works for a second batch

show_doc(Learner.all_batches)

# +
#|hide
learn = synth_learner(n_train=5, cbs=VerboseCallback())
learn.opt = SGD(learn.parameters(), lr=learn.lr)
with redirect_stdout(io.StringIO()): 
    learn(_before_epoch)
    learn.epoch,learn.dl = 0,learn.dls.train
    learn('before_train')
test_stdout(learn.all_batches, '\n'.join(batch_events * 5))
test_eq(learn.train_iter, 5)

valid_events = ['before_batch', 'after_pred', 'after_loss', 'after_batch']
with redirect_stdout(io.StringIO()): 
    learn.dl = learn.dls.valid
    learn('before_validate')
test_stdout(learn.all_batches, '\n'.join(valid_events * 2))
test_eq(learn.train_iter, 5)
# -

#|hide
learn = synth_learner(n_train=5, cbs=VerboseCallback())
test_stdout(lambda: learn(_before_epoch), 'before_fit\nbefore_epoch')
test_eq(learn.loss, tensor(0.))

#|hide
learn.opt = SGD(learn.parameters(), lr=learn.lr)
learn.epoch = 0
test_stdout(lambda: learn._do_epoch_train(), '\n'.join(['before_train'] + batch_events * 5 + ['after_train']))

#|hide
test_stdout(learn._do_epoch_validate, '\n'.join(['before_validate'] + valid_events * 2+ ['after_validate']))

show_doc(Learner.create_opt)

# This method is called internally to create the optimizer, the hyper-parameters are then adjusted by what you pass to `Learner.fit` or your particular schedulers (see `callback.schedule`).

learn = synth_learner(n_train=5, cbs=VerboseCallback())
assert learn.opt is None
learn.create_opt()
assert learn.opt is not None
test_eq(learn.opt.hypers[0]['lr'], learn.lr)

learn = synth_learner(n_train=5, cbs=VerboseCallback(), opt_func=partial(OptimWrapper, opt=torch.optim.Adam))
assert learn.opt is None
learn.create_opt()
assert learn.opt is not None
test_eq(learn.opt.hypers[0]['lr'], learn.lr)

wrapper_lr = 1
learn = synth_learner(n_train=5, cbs=VerboseCallback(), opt_func=partial(OptimWrapper, opt=torch.optim.Adam, lr=wrapper_lr))
assert learn.opt is None
learn.create_opt()
assert learn.opt is not None
test_eq(learn.opt.hypers[0]['lr'], wrapper_lr)


# ### Callback handling

# We only describe the basic functionality linked to `Callback`s here. To learn more about `Callback`s and how to write them, check the [callback.core](http://docs.fast.ai/callback.core.html) module documentation.
#
# Let's first see how the `Callback`s become attributes of `Learner`:

# +
#Test init with callbacks
class TstCallback(Callback):
    def batch_begin(self): self.learn.a = self.a + 1

tst_learn = synth_learner()
test_eq(len(tst_learn.cbs), 1)
assert hasattr(tst_learn, ('train_eval'))

tst_learn = synth_learner(cbs=TstCallback())
test_eq(len(tst_learn.cbs), 2)
assert hasattr(tst_learn, ('tst'))
# -

show_doc(Learner.__call__)

# This how the `Callback`s are called internally. For instance a `VerboseCallback` just prints the event names (can be useful for debugging):

learn = synth_learner(cbs=VerboseCallback())
learn('after_fit')

show_doc(Learner.add_cb)

learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
test_eq(len(learn.cbs), 2)
assert isinstance(learn.cbs[1], TestTrainEvalCallback)
test_eq(learn.train_eval.learn, learn)

show_doc(Learner.add_cbs)

learn.add_cbs([TestTrainEvalCallback(), TestTrainEvalCallback()])
test_eq(len(learn.cbs), 4)

show_doc(Learner.added_cbs)

learn = synth_learner()
test_eq(len(learn.cbs), 1)
with learn.added_cbs(TestTrainEvalCallback()):
    test_eq(len(learn.cbs), 2)

show_doc(Learner.ordered_cbs)

# By order, we mean using the internal ordering of the `Callback`s (see `callback.core` for more information on how it works).

learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
learn.ordered_cbs('before_fit')

show_doc(Learner.remove_cb)

learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
cb = learn.cbs[1]
learn.remove_cb(learn.cbs[1])
test_eq(len(learn.cbs), 1)
assert cb.learn is None
assert not getattr(learn,'test_train_eval',None)

# `cb` can simply be the class of the `Callback` we want to remove (in which case all instances of that callback are removed).

learn = synth_learner()
learn.add_cbs([TestTrainEvalCallback(), TestTrainEvalCallback()])
learn.remove_cb(TestTrainEvalCallback)
test_eq(len(learn.cbs), 1)
assert not getattr(learn,'test_train_eval',None)

show_doc(Learner.remove_cbs)

# Elements of `cbs` can either be types of callbacks or actual callbacks of the `Learner`.

learn = synth_learner()
learn.add_cbs([TestTrainEvalCallback() for _ in range(3)])
cb = learn.cbs[1]
learn.remove_cbs(learn.cbs[1:])
test_eq(len(learn.cbs), 1)

show_doc(Learner.removed_cbs)

# Elements of `cbs` can either be types of callbacks or actual callbacks of the `Learner`.

learn = synth_learner()
learn.add_cb(TestTrainEvalCallback())
with learn.removed_cbs(learn.cbs[1]):
    test_eq(len(learn.cbs), 1)
test_eq(len(learn.cbs), 2)

show_doc(Learner.show_training_loop)

# At each step, callbacks are shown in order, which can help debugging.

learn = synth_learner()
learn.show_training_loop()


#|export
def _before_batch_cb(f, self):
    xb,yb = f(self, self.xb, self.yb)
    self.learn.xb,self.learn.yb = xb,yb


#|export
def before_batch_cb(f):
    "Shortcut for creating a Callback on the `before_batch` event, which takes and returns `xb,yb`"
    return Callback(before_batch=partial(_before_batch_cb, f))


# In order to change the data passed to your model, you will generally want to hook into the `before_batch` event, like so:

class TstCallback(Callback):
    def before_batch(self):
        self.learn.xb = self.xb + 1000
        self.learn.yb = self.yb - 1000


# Since that is so common, we provide the `before_batch_cb` decorator to make it easier.

@before_batch_cb
def cb(self, xb, yb): return xb+1000,yb-1000


# +
#|hide
# test SkipToEpoch callback
class TestSkipToEpoch(Callback):
    def after_train(self):
        assert self.epoch >= 2
learn = synth_learner(cbs=TestSkipToEpoch())
learn.fit(4, start_epoch=2)

learn = synth_learner()
p0_pre = first(learn.model.parameters()).data.clone()
learn.fit(3, start_epoch=3)
p0 = first(learn.model.parameters()).data
test_eq(p0_pre, p0)


# -

# ### Serializing

#|export
@patch
@delegates(save_model)
def save(self:Learner, file, **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    save_model(file, self.model, getattr(self,'opt',None), **kwargs)
    return file


# `file` can be a `Path`, a `string` or a buffer. `pickle_protocol` is passed along to `torch.save`.

#|export
@patch
@delegates(load_model)
def load(self:Learner, file, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`"
    if device is None and hasattr(self.dls, 'device'): device = self.dls.device
    if self.opt is None: self.create_opt()
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    distrib_barrier()
    load_model(file, self.model, self.opt, device=device, **kwargs)
    return self


# `file` can be a `Path`, a `string` or a buffer. Use `device` to load the model/optimizer state on a device different from the one it was saved.

with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=d)
    learn.fit(1)
    
    #Test save created a file
    learn.save('tmp')
    assert (Path(d)/'models/tmp.pth').exists()
    
    #Test load did load the model
    learn1 = synth_learner(path=d)
    learn1 = learn1.load('tmp')
    test_eq(learn.a, learn1.a)
    test_eq(learn.b, learn1.b)
    test_eq(learn.opt.state_dict(), learn1.opt.state_dict())

#|hide
#Test load works when the model is saved without opt
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=d)
    learn.fit(1)
    learn.save('tmp', with_opt=False)
    learn1 = synth_learner(path=d)
    learn1 = learn1.load('tmp', with_opt=False)
    test_eq(learn.a, learn1.a)
    test_eq(learn.b, learn1.b)
    test_ne(learn.opt.state_dict(), learn1.opt.state_dict())


#|export
@patch
def export(self:Learner, fname='export.pkl', pickle_module=pickle, pickle_protocol=2):
    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib(): return # don't export if child proc
    self._end_cleanup()
    old_dbunch = self.dls
    self.dls = self.dls.new_empty()
    state = self.opt.state_dict() if self.opt is not None else None
    self.opt = None
    with warnings.catch_warnings():
        #To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(self, self.path/fname, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
    self.create_opt()
    if state is not None: self.opt.load_state_dict(state)
    self.dls = old_dbunch


# The `Learner` is saved in `self.path/fname`, using `pickle_protocol`. Note that serialization in Python saves the names of functions, not the code itself. Therefore, any custom code you have for models, data transformation, loss function etc... should be put in a module that you will import in your training environment before exporting, and in your deployment environment before loading it.

#|export
def load_learner(fname, cpu=True, pickle_module=pickle):
    "Load a `Learner` object in `fname`, by default putting it on the `cpu`"
    distrib_barrier()
    map_loc = 'cpu' if cpu else default_device()
    try: res = torch.load(fname, map_location=map_loc, pickle_module=pickle_module)
    except AttributeError as e: 
        e.args = [f"Custom classes or functions exported with your `Learner` not available in namespace.\Re-declare/import before loading:\n\t{e.args[0]}"]
        raise
    if cpu: 
        res.dls.cpu()
        if hasattr(res, 'channels_last'): res = res.to_contiguous(to_fp32=True)
        elif hasattr(res, 'mixed_precision'): res = res.to_fp32()
        elif hasattr(res, 'non_native_mixed_precision'): res = res.to_non_native_fp32()
    return res


# :::{.callout-warning}
#
# `load_learner` requires all your custom code be in the exact same place as when exporting your `Learner` (the main script, or the module you imported it from).
#
# :::

# ### DataLoader aware `to_detach` -

show_doc(Learner.to_detach)

# fastai provides `to_detach` which by default detachs tensor gradients, and gathers (calling `maybe_gather`) tensors from all ranks if running in distributed data parallel (DDP) mode.
#
# When running in DDP mode all ranks need to have the same batch size, and `DistributedDL` takes care of padding batches as needed; however when gathering all tensors (e.g. for calculating metrics, inference, etc.) we need to discard the padded items. `DistributedDL` provides a method `to_detach` that removes padding appropriately.
#
# Calling the learner's `to_detach` method will attempt to find a `to_detach` method in the learner's last used `DataLoader` `dl` and use that one if found, otherwise it will resort to the vanilla `to_detach`.

#|hide
learn = synth_learner()
test_eq(learn.to_detach(Tensor([123])),Tensor([123]))
learn.dl = learn.dls[0]
test_eq(learn.to_detach(Tensor([123])),Tensor([123]))
learn.dl.to_detach = lambda b,cpu,gather: b-100
test_eq(learn.to_detach(Tensor([123.])),Tensor([23.]))


# ## Metrics -

#|export
@docs
class Metric():
    "Blueprint for defining a metric"
    def reset(self): pass
    def accumulate(self, learn): pass
    @property
    def value(self): raise NotImplementedError

    @property
    def name(self): return class2attr(self, 'Metric')

    _docs = dict(
        reset="Reset inner state to prepare for new computation",
        name="Name of the `Metric`, camel-cased and with Metric removed",
        accumulate="Use `learn` to update the state with new results",
        value="The value of the metric")


show_doc(Metric, title_level=3)

# Metrics can be simple averages (like accuracy) but sometimes their computation is a little bit more complex and can't be averaged over batches (like precision or recall), which is why we need a special class for them. For simple functions that can be computed as averages over batches, we can use the class `AvgMetric`, otherwise you'll need to implement the following methods.
#
# :::{.callout-note}
#
# If your <code>Metric</code> has state depending on tensors, don't forget to store it on the CPU to avoid any potential memory leaks.
#
# :::

show_doc(Metric.reset)

show_doc(Metric.accumulate)

show_doc(Metric.value, name='Metric.value')

show_doc(Metric.name, name='Metric.name')


#|export
class AvgMetric(Metric):
    "Average the values of `func` taking into account potential different batch sizes"
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__


show_doc(AvgMetric, title_level=3)

learn = synth_learner()
tst = AvgMetric(lambda x,y: (x-y).abs().mean())
t,u = torch.randn(100),torch.randn(100)
tst.reset()
for i in range(0,100,25): 
    learn.pred,learn.yb = t[i:i+25],(u[i:i+25],)
    tst.accumulate(learn)
test_close(tst.value, (t-u).abs().mean())

#|hide
#With varying batch size
tst.reset()
splits = [0, 30, 50, 60, 100]
for i in range(len(splits )-1): 
    learn.pred,learn.yb = t[splits[i]:splits[i+1]],(u[splits[i]:splits[i+1]],)
    tst.accumulate(learn)
test_close(tst.value, (t-u).abs().mean())


#|export
class AvgLoss(Metric):
    "Average the losses taking into account potential different batch sizes"
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(learn.loss.mean())*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return "loss"


show_doc(AvgLoss, title_level=3)

tst = AvgLoss()
t = torch.randn(100)
tst.reset()
for i in range(0,100,25): 
    learn.yb,learn.loss = t[i:i+25],t[i:i+25].mean()
    tst.accumulate(learn)
test_close(tst.value, t.mean())

#|hide
#With varying batch size
tst.reset()
splits = [0, 30, 50, 60, 100]
for i in range(len(splits )-1): 
    learn.yb,learn.loss = t[splits[i]:splits[i+1]],t[splits[i]:splits[i+1]].mean()
    tst.accumulate(learn)
test_close(tst.value, t.mean())


#|export
class AvgSmoothLoss(Metric):
    "Smooth average of the losses (exponentially weighted with `beta`)"
    def __init__(self, beta=0.98): self.beta = beta
    def reset(self):               self.count,self.val = 0,tensor(0.)
    def accumulate(self, learn):
        self.count += 1
        self.val = torch.lerp(to_detach(learn.loss.mean()), self.val, self.beta)
    @property
    def value(self): return self.val/(1-self.beta**self.count)


show_doc(AvgSmoothLoss, title_level=3)

tst = AvgSmoothLoss()
t = torch.randn(100)
tst.reset()
val = tensor(0.)
for i in range(4): 
    learn.loss = t[i*25:(i+1)*25].mean()
    tst.accumulate(learn)
    val = val*0.98 + t[i*25:(i+1)*25].mean()*(1-0.98)
    test_close(val/(1-0.98**(i+1)), tst.value)


#|export
class ValueMetric(Metric):
    "Use to include a pre-calculated metric value (for instance calculated in a `Callback`) and returned by `func`"
    def __init__(self, func, metric_name=None): store_attr('func, metric_name')

    @property
    def value(self): return self.func()

    @property
    def name(self): return self.metric_name if self.metric_name else self.func.__name__


show_doc(ValueMetric, title_level=3)


# +
def metric_value_fn(): return 5e-3

vm = ValueMetric(metric_value_fn, 'custom_value_metric')
test_eq(vm.value, 5e-3)
test_eq(vm.name, 'custom_value_metric')

vm = ValueMetric(metric_value_fn)
test_eq(vm.name, 'metric_value_fn')
# -

# ## Recorder --

#|export
from fastprogress.fastprogress import format_time


#|export
def _maybe_item(t):
    t = t.value
    try: return t.item()
    except: return t


#|export
class Recorder(Callback):
    "Callback that registers statistics (lr, loss and metrics) during training"
    _stateattrs=('lrs','iters','losses','values')
    remove_on_fetch,order = True,50

    def __init__(self, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98):
        store_attr('add_time,train_metrics,valid_metrics')
        self.loss,self.smooth_loss = AvgLoss(),AvgSmoothLoss(beta=beta)

    def before_fit(self):
        "Prepare state for training"
        self.lrs,self.iters,self.losses,self.values = [],[],[],[]
        names = self.metrics.attrgot('name')
        if self.train_metrics and self.valid_metrics:
            names = L('loss') + names
            names = names.map('train_{}') + names.map('valid_{}')
        elif self.valid_metrics: names = L('train_loss', 'valid_loss') + names
        else: names = L('train_loss') + names
        if self.add_time: names.append('time')
        self.metric_names = 'epoch'+names
        self.smooth_loss.reset()

    def after_batch(self):
        "Update all metrics and records lr and smooth loss in training"
        if len(self.yb) == 0: return
        mets = self._train_mets if self.training else self._valid_mets
        for met in mets: met.accumulate(self.learn)
        if not self.training: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.smooth_loss.value)
        self.learn.smooth_loss = self.smooth_loss.value

    def before_epoch(self):
        "Set timer if `self.add_time=True`"
        self.cancel_train,self.cancel_valid = False,False
        if self.add_time: self.start_epoch = time.time()
        self.log = L(getattr(self, 'epoch', 0))

    def before_train   (self): self._train_mets[1:].map(Self.reset())
    def before_validate(self): self._valid_mets.map(Self.reset())
    def after_train   (self): self.log += self._train_mets.map(_maybe_item)
    def after_validate(self): self.log += self._valid_mets.map(_maybe_item)
    def after_cancel_train(self):    self.cancel_train = True
    def after_cancel_validate(self): self.cancel_valid = True

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learn.final_record = self.log[1:].copy()
        self.values.append(self.learn.final_record)
        if self.add_time: self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.smooth_loss.count)

    @property
    def _train_mets(self):
        if getattr(self, 'cancel_train', False): return L()
        return L(self.smooth_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_mets(self):
        if getattr(self, 'cancel_valid', False): return L()
        return (L(self.loss) + self.metrics if self.valid_metrics else L())

    def plot_loss(self, skip_start=5, with_valid=True, log=False, show_epochs=False, ax=None):
        if not ax:
            ax=plt.gca()
        if log:
            ax.loglog(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
        else:
            ax.plot(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
        if show_epochs:
            for x in self.iters:
                ax.axvline(x, color='grey', ls=':')
        ax.set_ylabel('loss')
        ax.set_xlabel('steps')
        ax.set_title('learning curve')
        if with_valid:
            idx = (np.array(self.iters)<skip_start).sum()
            valid_col = self.metric_names.index('valid_loss') - 1 
            ax.plot(self.iters[idx:], L(self.values[idx:]).itemgot(valid_col), label='valid')
            ax.legend()
        return ax


# +
#|export
add_docs(Recorder,
         before_train = "Reset loss and metrics state",
         after_train = "Log loss and metric values on the training set (if `self.training_metrics=True`)",
         before_validate = "Reset loss and metrics state",
         after_validate = "Log loss and metric values on the validation set",
         after_cancel_train = "Ignore training metrics for this epoch",
         after_cancel_validate = "Ignore validation metrics for this epoch",
         plot_loss = "Plot the losses from `skip_start` and onward. Optionally `log=True` for logarithmic axis, `show_epochs=True` for indicate epochs and a matplotlib axis `ax` to plot on.")

if Recorder not in defaults.callbacks: defaults.callbacks.append(Recorder)


# -

# By default, metrics are computed on the validation set only, although that can be changed by adjusting `train_metrics` and `valid_metrics`. `beta` is the weight used to compute the exponentially weighted average of the losses (which gives the `smooth_loss` attribute to `Learner`).
#
# The `logger` attribute of a `Learner` determines what happens to those metrics. By default, it just print them:

#Test printed output
def tst_metric(out, targ): return F.mse_loss(out, targ)
learn = synth_learner(n_train=5, metrics=tst_metric, default_cbs=False, cbs=[TrainEvalCallback, Recorder])
# pat = r"[tensor\(\d.\d*\), tensor\(\d.\d*\), tensor\(\d.\d*\), 'dd:dd']"
pat = r"\[\d, \d+.\d+, \d+.\d+, \d+.\d+, '\d\d:\d\d'\]"
test_stdout(lambda: learn.fit(1), pat, regex=True)


#|hide
class TestRecorderCallback(Callback):
    order=51
    def before_fit(self): 
        self.train_metrics,self.add_time = self.recorder.train_metrics,self.recorder.add_time
        self.beta = self.recorder.smooth_loss.beta
        for m in self.metrics: assert isinstance(m, Metric)
        test_eq(self.recorder.smooth_loss.val, 0.)
        #To test what the recorder logs, we use a custom logger function.
        self.learn.logger = self.test_log
        self.old_smooth,self.count = tensor(0.),0
    
    def after_batch(self):
        if self.training:
            self.count += 1
            test_eq(len(self.recorder.lrs), self.count)
            test_eq(self.recorder.lrs[-1], self.opt.hypers[-1]['lr'])
            test_eq(len(self.recorder.losses), self.count)
            smooth = (1 - self.beta**(self.count-1)) * self.old_smooth * self.beta + self.loss * (1-self.beta)
            smooth /= 1 - self.beta**self.count
            test_close(self.recorder.losses[-1], smooth, eps=1e-4)
            test_close(self.smooth_loss, smooth, eps=1e-4)
            self.old_smooth = self.smooth_loss
        self.bs += find_bs(self.yb)
        if not self.training: test_eq(self.recorder.loss.count, self.bs)
        if self.train_metrics or not self.training: 
            for m in self.metrics: test_eq(m.count, self.bs)
        self.losses.append(self.loss.detach().cpu())
    
    def before_epoch(self): 
        if self.add_time: self.start_epoch = time.time()
        self.log = [self.epoch]

    def before_train(self):
        self.bs = 0
        self.losses = []
        for m in self.recorder._train_mets: test_eq(m.count, self.bs)
            
    def after_train(self):
        mean = tensor(self.losses).mean()
        self.log += [self.smooth_loss, mean] if self.train_metrics else [self.smooth_loss]
        test_close(self.log, self.recorder.log)
        self.losses = []
    
    def before_validate(self):
        self.bs = 0
        self.losses = []
        for m in [self.recorder.loss] + self.metrics: test_eq(m.count, self.bs)
    
    def test_log(self, log):
        res = tensor(self.losses).mean()
        self.log += [res, res]
        if self.add_time: self.log.append(format_time(time.time() - self.start_epoch))
        test_close(log[:-1], self.log[:-1])
        test_eq(log[-1], self.log[-1])


# +
#|hide
def _get_learn():
    return synth_learner(n_train=5, metrics = tst_metric, default_cbs=False, cbs=[TrainEvalCallback, Recorder, TestRecorderCallback])

learn = _get_learn()
learn.fit(1)
test_eq(learn.recorder.metric_names, ['epoch', 'train_loss', 'valid_loss', 'tst_metric', 'time'])

learn = _get_learn()
learn.recorder.train_metrics=True
learn.fit(1)
test_eq(learn.recorder.metric_names, 
        ['epoch', 'train_loss', 'train_tst_metric', 'valid_loss', 'valid_tst_metric', 'time'])

learn = _get_learn()
learn.recorder.add_time=False
learn.fit(1)
test_eq(learn.recorder.metric_names, ['epoch', 'train_loss', 'valid_loss', 'tst_metric'])


# -

#|hide
#Test numpy metric
def tst_metric_np(out, targ): return F.mse_loss(out, targ).numpy()
learn = synth_learner(n_train=5, metrics=tst_metric_np)
learn.fit(1)

# ### Internals

show_doc(Recorder.before_fit)

show_doc(Recorder.before_epoch)

show_doc(Recorder.before_validate)

show_doc(Recorder.after_batch)

show_doc(Recorder.after_epoch)

# ### Plotting tools

show_doc(Recorder.plot_loss)

#|hide
learn.recorder.plot_loss(skip_start=1)


# ## CastToTensor -

#|exporti
def _cast_tensor(x): 
    if isinstance(x, tuple): return tuple(_cast_tensor(x_) for x_ in x)
    else: return cast(x, Tensor) if isinstance(x,torch.Tensor) else x


#|export
class CastToTensor(Callback):
    "Cast Subclassed Tensors to `Tensor`"
    order=9 # Right before MixedPrecision

    def before_batch(self):
        self.learn.xb,self.learn.yb = _cast_tensor(self.learn.xb),_cast_tensor(self.learn.yb)


# Workaround for bug in PyTorch where subclassed tensors, such as `TensorBase`, train up to ~20% slower than `Tensor` when passed to a model. Added to `Learner` by default.
#
# CastToTensor's order is right before `MixedPrecision` so callbacks which make use of fastai's tensor subclasses still can use them.
#
# If inputs are not a subclassed tensor or tuple of tensors, you may need to cast inputs in `Learner.xb` and `Learner.yb` to `Tensor` via your own callback or in the dataloader before `Learner` performs the forward pass.
#
# If the CastToTensor workaround interferes with custom code, it can be removed:
#
# ```python
# learn = Learner(...)
# learn.remove_cb(CastToTensor)
# ```
#
# You should verify your inputs are of type `Tensor` or implement a cast to `Tensor` via a custom callback or dataloader if CastToTensor is removed.

#|export
if CastToTensor not in defaults.callbacks: defaults.callbacks.append(CastToTensor)

# ## Inference functions

show_doc(Learner.validate)

#Test result
learn = synth_learner(n_train=5, metrics=tst_metric)
res = learn.validate()
test_eq(res[0], res[1])
x,y = learn.dls.valid_ds.tensors
test_close(res[0], F.mse_loss(learn.model(x), y), 1e-3)

#|hide
#Test other dl
res = learn.validate(dl=learn.dls.train)
test_eq(res[0], res[1])
x,y = learn.dls.train_ds.tensors
test_close(res[0], F.mse_loss(learn.model(x), y), 1e-3)

show_doc(Learner.get_preds)

# `with_decoded` will also return the decoded predictions using the <code>decodes</code> function of the loss function (if it exists). For instance, fastai's `CrossEntropyFlat` takes the argmax or predictions in its decodes. 
#
# Depending on the `loss_func` attribute of `Learner`, an activation function will be picked automatically so that the predictions make sense. For instance if the loss is a case of cross-entropy, a softmax will be applied, or if the loss is binary cross entropy with logits, a sigmoid will be applied. If you want to make sure a certain activation function is applied, you can pass it with `act`.
#
# `save_preds` and `save_targs` should be used when your predictions are too big to fit all in memory. Give a `Path` object that points to a folder where the predictions and targets will be saved.
#
# `concat_dim` is the batch dimension, where all the tensors will be concatenated.
#
# `inner` is an internal attribute that tells `get_preds` it's called internally, inside another training loop, to avoid recursion errors.

# :::{.callout-note}
#
# If you want to use the option `with_loss=True` on a custom loss function, make sure you have implemented a `reduction` attribute that supports 'none'
#
# :::

# +
#Test result
learn = synth_learner(n_train=5, metrics=tst_metric)
preds,targs = learn.get_preds()
x,y = learn.dls.valid_ds.tensors
test_eq(targs, y)
test_close(preds, learn.model(x))

preds,targs = learn.get_preds(act = torch.sigmoid)
test_eq(targs, y)
test_close(preds, torch.sigmoid(learn.model(x)))
# -

#|hide
#Test get_preds work with ds not evenly divisible by bs
learn = synth_learner(n_train=2.5, metrics=tst_metric)
preds,targs = learn.get_preds(ds_idx=0)
#Also make sure this version works when drop_last of the dl would be True
preds,targs = learn.get_preds(dl=learn.dls.train)

# +
#|hide
#Test other dataset
x = torch.randn(16*5)
y = 2*x + 3 + 0.1*torch.randn(16*5)
dl = TfmdDL(TensorDataset(x, y), bs=16)
preds,targs = learn.get_preds(dl=dl)
test_eq(targs, y)
test_close(preds, learn.model(x))

#Test with loss
preds,targs,losses = learn.get_preds(dl=dl, with_loss=True)
test_eq(targs, y)
test_close(preds, learn.model(x))
test_close(losses, F.mse_loss(preds, targs, reduction='none'))

#Test with inputs
inps,preds,targs = learn.get_preds(dl=dl, with_input=True)
test_eq(inps,x)
test_eq(targs, y)
test_close(preds, learn.model(x))
# -

#|hide
#Test with no target
learn = synth_learner(n_train=5)
x = torch.randn(16*5)
dl = TfmdDL(TensorDataset(x), bs=16)
preds,targs = learn.get_preds(dl=dl)
assert targs is None


# +
#|hide
#Test with targets that are tuples
def _fake_loss(x,y,z,reduction=None): return F.mse_loss(x,y)

learn = synth_learner(n_train=5)
x = torch.randn(16*5)
y = 2*x + 3 + 0.1*torch.randn(16*5)
learn.dls.n_inp=1
learn.loss_func = _fake_loss
dl = TfmdDL(TensorDataset(x, y, y), bs=16)
preds,targs = learn.get_preds(dl=dl)
test_eq(targs, [y,y])


# +
#|hide
#Test with inputs that are tuples
class _TupleModel(Module):
    def __init__(self, model): self.model=model
    def forward(self, x1, x2): return self.model(x1)

learn = synth_learner(n_train=5)
#learn.dls.n_inp=2
x = torch.randn(16*5)
y = 2*x + 3 + 0.1*torch.randn(16*5)
learn.model = _TupleModel(learn.model)
learn.dls = DataLoaders(TfmdDL(TensorDataset(x, x, y), bs=16),TfmdDL(TensorDataset(x, x, y), bs=16))
inps,preds,targs = learn.get_preds(ds_idx=0, with_input=True)
test_eq(inps, [x,x])
t = learn.get_preds(ds_idx=0, with_input=True)
# -

#|hide
#Test auto activation function is picked
learn = synth_learner(n_train=5)
learn.loss_func = BCEWithLogitsLossFlat()
x = torch.randn(16*5)
y = 2*x + 3 + 0.1*torch.randn(16*5)
dl = TfmdDL(TensorDataset(x, y), bs=16)
preds,targs = learn.get_preds(dl=dl)
test_close(preds, torch.sigmoid(learn.model(x)))

#|hide
#Test reorder is done
learn = synth_learner(n_train=5)
x = torch.randn(16*5)
y = 2*x + 3 + 0.1*torch.randn(16*5)
dl = TfmdDL(TensorDataset(x, y), bs=16, shuffle=True)
preds,targs = learn.get_preds(dl=dl)
test_eq(targs, y)

#|hide
inps,preds,targs = learn.get_preds(ds_idx=0, with_input=True)
tst = learn.get_preds(ds_idx=0, with_input=True, with_decoded=True)

show_doc(Learner.predict)


# It returns a tuple of three elements with, in reverse order,
# - the prediction from the model, potentially passed through the activation of the loss function (if it has one)
# - the decoded prediction, using the potential <code>decodes</code> method from it
# - the fully decoded prediction, using the transforms used to build the `Datasets`/`DataLoaders`

# `rm_type_tfms` is a deprecated argument that should not be used and will be removed in a future version. `with_input` will add the decoded inputs to the result.

# +
class _FakeLossFunc(Module):
    reduction = 'none'
    def forward(self, x, y): return F.mse_loss(x,y)
    def activation(self, x): return x+1
    def decodes(self, x):    return 2*x

class _Add1(Transform):
    def encodes(self, x): return x+1
    def decodes(self, x): return x-1
    
learn = synth_learner(n_train=5)
dl = TfmdDL(Datasets(torch.arange(50), tfms = [L(), [_Add1()]]))
learn.dls = DataLoaders(dl, dl)
learn.loss_func = _FakeLossFunc()

inp = tensor([2.])
out = learn.model(inp).detach()+1  #applying model + activation
dec = 2*out                        #decodes from loss function
full_dec = dec-1                   #decodes from _Add1
test_eq(learn.predict(inp), [full_dec,dec,out])
test_eq(learn.predict(inp, with_input=True), [inp,full_dec,dec,out])
# -

show_doc(Learner.show_results)

# Will show `max_n` samples (unless the batch size of `ds_idx` or `dl` is less than `max_n`, in which case it will show as many samples) and `shuffle` the data unless you pass `false` to that flag. `kwargs` are application-dependent.
#
# We can't show an example on our synthetic `Learner`, but check all the beginners tutorials which will show you how that method works across applications.

# The last functions in this section are used internally for inference, but should be less useful to you.

show_doc(Learner.no_logging)

learn = synth_learner(n_train=5, metrics=tst_metric)
with learn.no_logging():
    test_stdout(lambda: learn.fit(1), '')
test_eq(learn.logger, print)

show_doc(Learner.loss_not_reduced)

# This requires your loss function to either have a `reduction` attribute or a `reduction` argument (like all fastai and PyTorch loss functions).

#|hide
test_eq(learn.loss_func.reduction, 'mean')
with learn.loss_not_reduced():
    test_eq(learn.loss_func.reduction, 'none')
    x,y = learn.dls.one_batch()
    p = learn.model(x)
    losses = learn.loss_func(p, y)
    test_eq(losses.shape, y.shape)
    test_eq(losses, F.mse_loss(p,y, reduction='none'))
test_eq(learn.loss_func.reduction, 'mean')


# ## Transfer learning

# +
#|export
@patch
def freeze_to(self:Learner, n):
    if self.opt is None: self.create_opt()
    self.opt.freeze_to(n)
    self.opt.clear_state()

@patch
def freeze(self:Learner): self.freeze_to(-1)

@patch
def unfreeze(self:Learner): self.freeze_to(0)

add_docs(Learner,
         freeze_to="Freeze parameter groups up to `n`",
         freeze="Freeze up to last parameter group",
         unfreeze="Unfreeze the entire model")


# +
#|hide
class _TstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
        self.tst = nn.Sequential(nn.Linear(4,5), nn.BatchNorm1d(3))
        self.tst[0].bias.data,self.tst[1].bias.data = torch.randn(5),torch.randn(3) 
    def forward(self, x): return x * self.a + self.b
    
class _PutGrad(Callback):
    def before_step(self):
        for p in self.learn.tst.parameters():
            if p.requires_grad: p.grad = torch.ones_like(p.data)

def _splitter(m): return [list(m.tst[0].parameters()), list(m.tst[1].parameters()), [m.a,m.b]]
            
learn = synth_learner(n_train=5, opt_func = partial(SGD), cbs=_PutGrad, splitter=_splitter, lr=1e-2)
learn.model = _TstModel()
learn.freeze()
init = [p.clone() for p in learn.tst.parameters()]
learn.fit(1, wd=0.)
end = list(learn.tst.parameters())
#linear was not trained
for i in [0,1]: test_close(end[i],init[i])
#bn was trained even frozen since `train_bn=True` by default
for i in [2,3]: test_close(end[i]-init[i], -0.05 * torch.ones_like(end[i]))

# +
#|hide
learn = synth_learner(n_train=5, opt_func = partial(SGD), cbs=_PutGrad, splitter=_splitter, train_bn=False, lr=1e-2)
learn.model = _TstModel()
learn.freeze()
init = [p.clone() for p in learn.tst.parameters()]
learn.fit(1, wd=0.)
end = list(learn.tst.parameters())
#linear and bn were not trained
for i in range(4): test_close(end[i],init[i])

learn.freeze_to(-2)
init = [p.clone() for p in learn.tst.parameters()]
learn.fit(1, wd=0.)
end = list(learn.tst.parameters())
#linear was not trained
for i in [0,1]: test_close(end[i],init[i])
#bn was trained 
for i in [2,3]: test_close(end[i]-init[i], -0.05 * torch.ones_like(end[i]))
    
learn.unfreeze()
init = [p.clone() for p in learn.tst.parameters()]
learn.fit(1, wd=0.)
end = list(learn.tst.parameters())
#linear and bn were trained
for i in range(4): test_close(end[i]-init[i], -0.05 * torch.ones_like(end[i]), 1e-3)


# -

# ## TTA

#|export
@patch
def tta(self:Learner, ds_idx=1, dl=None, n=4, item_tfms=None, batch_tfms=None, beta=0.25, use_max=False):
    "Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation"
    if dl is None: dl = self.dls[ds_idx].new(shuffled=False, drop_last=False)
    if item_tfms is not None or batch_tfms is not None: dl = dl.new(after_item=item_tfms, after_batch=batch_tfms)
    try:
        self(_before_epoch)
        with dl.dataset.set_split_idx(0), self.no_mbar():
            if hasattr(self,'progress'): self.progress.mbar = master_bar(list(range(n)))
            aug_preds = []
            for i in self.progress.mbar if hasattr(self,'progress') else range(n):
                self.epoch = i #To keep track of progress on mbar since the progress callback will use self.epoch
                aug_preds.append(self.get_preds(dl=dl, inner=True)[0][None])
        aug_preds = torch.cat(aug_preds)
        aug_preds = aug_preds.max(0)[0] if use_max else aug_preds.mean(0)
        self.epoch = n
        with dl.dataset.set_split_idx(1): preds,targs = self.get_preds(dl=dl, inner=True)
    finally: self(event.after_fit)

    if use_max: return torch.stack([preds, aug_preds], 0).max(0)[0],targs
    preds = (aug_preds,preds) if beta is None else torch.lerp(aug_preds, preds, beta)
    return preds,targs


# In practice, we get the predictions `n` times with the transforms of the training set and average those. The final predictions are `(1-beta)` multiplied by this average + `beta` multiplied by the predictions obtained with the transforms of the dataset. Set `beta` to `None` to get a tuple of the predictions and tta results. You can also use the maximum of all predictions instead of an average by setting `use_max=True`.
#
# If you want to use new transforms, you can pass them with `item_tfms` and `batch_tfms`.

#|hide
learn = synth_learner()
dl = TfmdDL(Datasets(torch.arange(50), [noop,noop]))
learn.dls = DataLoaders(dl, dl)
preds,targs = learn.tta()
assert len(preds),len(targs)

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


