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
#|default_exp callback.core
# -

#|export
from __future__ import annotations
from fastai.data.all import *
from fastai.optimizer import *
from fastai.losses import BaseLoss

#|hide
from nbdev.showdoc import *

#|export
_all_ = ['CancelStepException','CancelBackwardException','CancelFitException','CancelEpochException','CancelTrainException','CancelValidException','CancelBatchException']

# # Callbacks
#
# > Basic callbacks for Learner

# ## Events

# Callbacks can occur at any of these times:: *after_create before_fit before_epoch before_train before_batch after_pred after_loss before_backward after_cancel_backward after_backward before_step after_cancel_step after_step after_cancel_batch after_batch after_cancel_train after_train before_validate after_cancel_validate after_validate after_cancel_epoch after_epoch after_cancel_fit after_fit*.

# +
#|export
_events = L.split('after_create before_fit before_epoch before_train before_batch after_pred after_loss \
    before_backward after_cancel_backward after_backward before_step after_cancel_step after_step \
    after_cancel_batch after_batch after_cancel_train after_train before_validate after_cancel_validate \
    after_validate after_cancel_epoch after_epoch after_cancel_fit after_fit')

mk_class('event', **_events.map_dict(),
         doc="All possible events as attributes to get tab-completion and typo-proofing")
# -

#|export
_all_ = ['event']

show_doc(event, name='event', title_level=3)

# To ensure that you are referring to an event (that is, the name of one of the times when callbacks are called) that exists, and to get tab completion of event names, use `event`:

test_eq(event.before_step, 'before_step')

# ## Callback - 

#|export
_inner_loop = "before_batch after_pred after_loss before_backward after_cancel_backward after_backward before_step after_step after_cancel_batch after_batch".split()

# +
#|export
_ex_docs = dict(
    CancelBatchException="Skip the rest of this batch and go to `after_batch`",
    CancelTrainException="Skip the rest of the training part of the epoch and go to `after_train`",
    CancelValidException="Skip the rest of the validation part of the epoch and go to `after_validate`",
    CancelEpochException="Skip the rest of this epoch and go to `after_epoch`",
    CancelStepException ="Skip stepping the optimizer",
    CancelBackwardException="Skip the backward pass and go to `after_backward`",
    CancelFitException  ="Interrupts training and go to `after_fit`")

for c,d in _ex_docs.items(): mk_class(c,sup=Exception,doc=d)


# -

#|export
@funcs_kwargs(as_method=True)
class Callback(Stateful,GetAttr):
    "Basic class handling tweaks of the training loop by changing a `Learner` in various events"
    order,_default,learn,run,run_train,run_valid = 0,'learn',None,True,True,True
    _methods = _events

    def __init__(self, **kwargs): assert not kwargs, f'Passed unknown events: {kwargs}'
    def __repr__(self): return type(self).__name__

    def __call__(self, event_name):
        "Call `self.{event_name}` if it's defined"
        _run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
               (self.run_valid and not getattr(self, 'training', False)))
        res = None
        if self.run and _run: 
            try: res = getcallable(self, event_name)()
            except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
            except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
        if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
        return res

    def __setattr__(self, name, value):
        "Set an attribute for a `Callback`"
        if hasattr(self.learn,name):
            warn(f"You are shadowing an attribute ({name}) that exists in the learner. Use `self.learn.{name}` to avoid this")
        super().__setattr__(name, value)

    @property
    def name(self):
        "Name of the `Callback`, camel-cased and with '*Callback*' removed"
        return class2attr(self, 'Callback')


# The training loop is defined in `Learner` a bit below and consists in a minimal set of instructions: looping through the data we:
#
# - compute the output of the model from the input
# - calculate a loss between this output and the desired target
# - compute the gradients of this loss with respect to all the model parameters
# - update the parameters accordingly
# - zero all the gradients
#
# Any tweak of this training loop is defined in a `Callback` to avoid over-complicating the code of the training loop, and to make it easy to mix and match different techniques (since they'll be defined in different callbacks). A callback can implement actions on the following events:
#
# - `after_create`: called after the `Learner` is created
# - `before_fit`: called before starting training or inference, ideal for initial setup.
# - `before_epoch`: called at the beginning of each epoch, useful for any behavior you need to reset at each epoch.
# - `before_train`: called at the beginning of the training part of an epoch.
# - `before_batch`: called at the beginning of each batch, just after drawing said batch. It can be used to do any setup necessary for the batch (like hyper-parameter scheduling) or to change the input/target before it goes in the model (change of the input with techniques like mixup for instance).
# - `after_pred`: called after computing the output of the model on the batch. It can be used to change that output before it's fed to the loss.
# - `after_loss`: called after the loss has been computed, but before the backward pass. It can be used to add any penalty to the loss (AR or TAR in RNN training for instance).
# - `before_backward`: called after the loss has been computed, but only in training mode (i.e. when the backward pass will be used)
# - `after_backward`: called after the backward pass, but before the update of the parameters. Generally `before_step` should be used instead.
# - `before_step`: called after the backward pass, but before the update of the parameters. It can be used to do any change to the gradients before said update (gradient clipping for instance).
# - `after_step`: called after the step and before the gradients are zeroed.
# - `after_batch`: called at the end of a batch, for any clean-up before the next one.
# - `after_train`: called at the end of the training phase of an epoch.
# - `before_validate`: called at the beginning of the validation phase of an epoch, useful for any setup needed specifically for validation.
# - `after_validate`: called at the end of the validation part of an epoch.
# - `after_epoch`: called at the end of an epoch, for any clean-up before the next one.
# - `after_fit`: called at the end of training, for final clean-up.

show_doc(Callback.__call__)


# One way to define callbacks is through subclassing:

class _T(Callback):
    def call_me(self): return "maybe"
test_eq(_T()("call_me"), "maybe")


# Another way is by passing the callback function to the constructor:

def cb(self): return "maybe"
_t = Callback(before_fit=cb)
test_eq(_t(event.before_fit), "maybe")

# `Callback`s provide a shortcut to avoid having to write `self.learn.bla` for any `bla` attribute we seek; instead, just write `self.bla`. This only works for getting attributes, *not* for setting them.

# +
mk_class('TstLearner', 'a')

class TstCallback(Callback):
    def batch_begin(self): print(self.a)

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
test_stdout(lambda: cb('batch_begin'), "1")
# -

# If you want to change the value of an attribute, you have to use `self.learn.bla`, no `self.bla`. In the example below, `self.a += 1` creates an `a` attribute of 2 in the callback instead of setting the `a` of the learner to 2. It also issues a warning that something is probably wrong:

learn.a


# +
class TstCallback(Callback):
    def batch_begin(self): self.a += 1

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
cb('batch_begin')
test_eq(cb.a, 2)
test_eq(cb.learn.a, 1)


# -

# A proper version needs to write `self.learn.a = self.a + 1`:

# +
class TstCallback(Callback):
    def batch_begin(self): self.learn.a = self.a + 1

learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
cb('batch_begin')
test_eq(cb.learn.a, 2)


# -

#|hide
class TstCallback(Callback):
    def batch_begin(self): self.learn.a = 1 + "a"
learn,cb = TstLearner(1),TstCallback()
cb.learn = learn
with ExceptionExpected(TypeError, regex=" in `TstCallback` when calling event `batch_begin`"):
    cb('batch_begin')

show_doc(Callback.name, name='Callback.name')

test_eq(TstCallback().name, 'tst')
class ComplicatedNameCallback(Callback): pass
test_eq(ComplicatedNameCallback().name, 'complicated_name')


# ## TrainEvalCallback -

#|export
class TrainEvalCallback(Callback):
    "`Callback` that tracks the number of iterations done and properly sets training/eval mode"
    order,run_valid = -10,False
    def after_create(self): self.learn.n_epoch = 1

    def before_fit(self):
        "Set the iter and epoch counters to 0, put the model and the right device"
        self.learn.epoch,self.learn.loss = 0,tensor(0.)
        self.learn.train_iter,self.learn.pct_train = 0,0.
        device = getattr(self.dls, 'device', default_device())
        self.model.to(device)
        if isinstance(self.loss_func, (nn.Module, BaseLoss)): self.loss_func.to(device)
        if hasattr(self.model, 'reset'): self.model.reset()

    def after_batch(self):
        "Update the iter counter (in training mode)"
        self.learn.pct_train += 1./(self.n_iter*self.n_epoch)
        self.learn.train_iter += 1

    def before_train(self):
        "Set the model to training mode"
        self.learn.pct_train=self.epoch/self.n_epoch
        self.model.train()
        self.learn.training=True

    def before_validate(self):
        "Set the model to validation mode"
        self.model.eval()
        self.learn.training=False


show_doc(TrainEvalCallback, title_level=3)

# This `Callback` is automatically added in every `Learner` at initialization.

# +
#|hide
#test of the TrainEvalCallback below in Learner.fit
# -

#|export
if not hasattr(defaults, 'callbacks'): defaults.callbacks = [TrainEvalCallback]

# ## Attributes available to callbacks

# When writing a callback, the following attributes of `Learner` are available:
#
# - `model`: the model used for training/validation
# - `dls`: the underlying `DataLoaders`
# - `loss_func`: the loss function used
# - `opt`: the optimizer used to update the model parameters
# - `opt_func`: the function used to create the optimizer
# - `cbs`: the list containing all `Callback`s
# - `dl`: current `DataLoader` used for iteration
# - `x`/`xb`: last input drawn from `self.dl` (potentially modified by callbacks). `xb` is always a tuple (potentially with one element) and `x` is detuplified. You can only assign to `xb`.
# - `y`/`yb`: last target drawn from `self.dl` (potentially modified by callbacks). `yb` is always a tuple (potentially with one element) and `y` is detuplified. You can only assign to `yb`.
# - `pred`: last predictions from `self.model` (potentially modified by callbacks)
# - `loss_grad`: last computed loss (potentially modified by callbacks)
# - `loss`: clone of `loss_grad` used for logging
# - `n_epoch`: the number of epochs in this training
# - `n_iter`: the number of iterations in the current `self.dl`
# - `epoch`: the current epoch index (from 0 to `n_epoch-1`)
# - `iter`: the current iteration index in `self.dl` (from 0 to `n_iter-1`)
#
# The following attributes are added by `TrainEvalCallback` and should be available unless you went out of your way to remove that callback:
#
# - `train_iter`: the number of training iterations done since the beginning of this training
# - `pct_train`: from 0. to 1., the percentage of training iterations completed
# - `training`:  flag to indicate if we're in training mode or not
#
# The following attribute is added by `Recorder` and should be available unless you went out of your way to remove that callback:
#
# - `smooth_loss`: an exponentially-averaged version of the training loss

# ## Callbacks control flow

# It happens that we may want to skip some of the steps of the training loop: in gradient accumulation, we don't always want to do the step/zeroing of the grads for instance. During an LR finder test, we don't want to do the validation phase of an epoch. Or if we're training with a strategy of early stopping, we want to be able to completely interrupt the training loop.
#
# This is made possible by raising specific exceptions the training loop will look for (and properly catch).

show_doc(CancelStepException, title_level=3)

show_doc(CancelBatchException, title_level=3)

show_doc(CancelBackwardException, title_level=3)

show_doc(CancelTrainException, title_level=3)

show_doc(CancelValidException, title_level=3)

show_doc(CancelEpochException, title_level=3)

show_doc(CancelFitException, title_level=3)


# You can detect one of those exceptions occurred and add code that executes right after with the following events:
#
# - `after_cancel_batch`: reached immediately after a `CancelBatchException` before proceeding to `after_batch`
# - `after_cancel_train`: reached immediately after a `CancelTrainException` before proceeding to `after_epoch`
# - `after_cancel_valid`: reached immediately after a `CancelValidException` before proceeding to `after_epoch`
# - `after_cancel_epoch`: reached immediately after a `CancelEpochException` before proceeding to `after_epoch`
# - `after_cancel_fit`: reached immediately after a `CancelFitException` before proceeding to `after_fit`

# ## Gather and fetch preds callbacks -

#|export
class GatherPredsCallback(Callback):
    "`Callback` that returns all predictions and targets, optionally `with_input` or `with_loss`"
    _stateattrs=('preds','targets','inputs','losses')
    def __init__(self,
        with_input:bool=False, # Whether to return inputs
        with_loss:bool=False, # Whether to return losses
        save_preds:Path=None, # Path to save predictions
        save_targs:Path=None, # Path to save targets
        with_preds:bool=True, # Whether to return predictions
        with_targs:bool=True, # Whether to return targets
        concat_dim:int=0, # Dimension to concatenate returned tensors
        pickle_protocol:int=2 # Pickle protocol used to save predictions and targets
    ):
        store_attr()

    def before_batch(self):
        "If `with_input`, detach batch inputs"
        if self.with_input: self.inputs.append((self.learn.to_detach(self.xb)))

    def before_validate(self):
        "Initialize containers"
        self.preds,self.targets = [],[]
        if self.with_input: self.inputs = []
        if self.with_loss:  self.losses = []

    def after_batch(self):
        "Save predictions, targets and potentially losses"
        if not hasattr(self, 'pred'): return
        preds,targs = self.learn.to_detach(self.pred),self.learn.to_detach(self.yb)
        if self.with_preds: self.preds.append(preds)
        if self.with_targs: self.targets.append(targs)
        if self.save_preds is not None: 
            torch.save(preds, self.save_preds/str(self.iter), pickle_protocol=self.pickle_protocol)
        if self.save_targs is not None: 
            torch.save(targs[0], self.save_targs/str(self.iter), pickle_protocol=self.pickle_protocol)
        if self.with_loss:
            bs = find_bs(self.yb)
            loss = self.loss if self.loss.numel() == bs else self.loss.view(bs,-1).mean(1)
            self.losses.append(self.learn.to_detach(loss))

    def after_validate(self):
        "Concatenate all recorded tensors"
        if not hasattr(self, 'preds'): return
        if self.with_input: self.inputs  = detuplify(to_concat(self.inputs, dim=self.concat_dim))
        if self.with_preds: self.preds   = detuplify(to_concat(self.preds, dim=self.concat_dim))
        if self.with_targs: self.targets = detuplify(to_concat(self.targets, dim=self.concat_dim))
        if self.with_loss:  self.losses  = to_concat(self.losses)

    def all_tensors(self) -> (Tensor, list):
        "Returns all recorded tensors in the order [inputs, preds, targets, losses]"
        res = [self.preds if self.with_preds else None, self.targets if self.with_targs else None]
        if self.with_input: res = [self.inputs] + res
        if self.with_loss:  res.append(self.losses)
        return res


show_doc(GatherPredsCallback, title_level=3)


#|export
class FetchPredsCallback(Callback):
    "A callback to fetch predictions during the training loop"
    remove_on_fetch = True
    def __init__(self,
        ds_idx:int=1, # Index of dataset, 0 for train, 1 for valid, used if `dl` is not present
        dl:DataLoader=None, # `DataLoader` used for fetching `Learner` predictions
        with_input:bool=False, # Whether to return inputs in `GatherPredsCallback`
        with_decoded:bool=False, # Whether to return decoded predictions
        cbs:Callback|MutableSequence=None, # `Callback` to temporarily remove from `Learner`
        reorder:bool=True # Whether to sort prediction results
    ):
        self.cbs = L(cbs)
        store_attr('ds_idx,dl,with_input,with_decoded,reorder')

    def after_validate(self):
        "Fetch predictions from `Learner` without `self.cbs` and `remove_on_fetch` callbacks"
        to_rm = L(cb for cb in self.learn.cbs if getattr(cb, 'remove_on_fetch', False))
        with self.learn.removed_cbs(to_rm + self.cbs) as learn:
            self.preds = learn.get_preds(ds_idx=self.ds_idx, dl=self.dl,
                with_input=self.with_input, with_decoded=self.with_decoded, inner=True, reorder=self.reorder)


show_doc(FetchPredsCallback, title_level=3)

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


