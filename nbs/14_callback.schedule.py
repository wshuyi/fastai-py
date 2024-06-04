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
#|default_exp callback.schedule
# -

#|export
from __future__ import annotations
from fastai.basics import *
from fastai.callback.tracker import SaveModelCallback

#|export
_all_ = ['SuggestionMethod']

#|hide
from nbdev.showdoc import *

# # Hyperparam schedule
#
# > Callback and helper functions to schedule any hyper-parameter

from fastai.test_utils import *


# ## Annealing

#|export
class _Annealer:
    def __init__(self, f, start, end): store_attr('f,start,end')
    def __call__(self, pos): return self.f(self.start, self.end, pos)


#|export
def annealer(f):
    "Decorator to make `f` return itself partially applied."
    @functools.wraps(f)
    def _inner(start, end): return _Annealer(f, start, end)
    return _inner


# This is the decorator we will use for all of our scheduling functions, as it transforms a function taking `(start, end, pos)` to something taking `(start, end)` and return a function depending of `pos`.

# +
#|export
#TODO Jeremy, make this pickle
#@annealer
#def SchedLin(start, end, pos): return start + pos*(end-start)
#@annealer
#def SchedCos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
#@annealer
#def SchedNo (start, end, pos): return start
#@annealer
#def SchedExp(start, end, pos): return start * (end/start) ** pos
#
#SchedLin.__doc__ = "Linear schedule function from `start` to `end`"
#SchedCos.__doc__ = "Cosine schedule function from `start` to `end`"
#SchedNo .__doc__ = "Constant schedule function with `start` value"
#SchedExp.__doc__ = "Exponential schedule function from `start` to `end`"

# +
#|export
def sched_lin(start, end, pos): return start + pos*(end-start)
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
def sched_no (start, end, pos): return start
def sched_exp(start, end, pos): return start * (end/start) ** pos

def SchedLin(start, end): return _Annealer(sched_lin, start, end)
def SchedCos(start, end): return _Annealer(sched_cos, start, end)
def SchedNo (start, end): return _Annealer(sched_no,  start, end)
def SchedExp(start, end): return _Annealer(sched_exp, start, end)

SchedLin.__doc__ = "Linear schedule function from `start` to `end`"
SchedCos.__doc__ = "Cosine schedule function from `start` to `end`"
SchedNo .__doc__ = "Constant schedule function with `start` value"
SchedExp.__doc__ = "Exponential schedule function from `start` to `end`"
# -

#|hide
tst = pickle.dumps(SchedCos(0, 5))

annealings = "NO LINEAR COS EXP".split()
p = torch.linspace(0.,1,100)
fns = [SchedNo, SchedLin, SchedCos, SchedExp]


#|export
def SchedPoly(start, end, power):
    "Polynomial schedule (of `power`) function from `start` to `end`"
    def _inner(pos): return start + (end - start) * pos ** power
    return _inner


for fn, t in zip(fns, annealings):
    plt.plot(p, [fn(2, 1e-2)(o) for o in p], label=t)
f = SchedPoly(2,1e-2,0.5)
plt.plot(p, [f(o) for o in p], label="POLY(0.5)")
plt.legend();

show_doc(SchedLin)

sched = SchedLin(0, 2)
test_eq(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.5, 1., 1.5, 2.])

show_doc(SchedCos)

sched = SchedCos(0, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.29289, 1., 1.70711, 2.])

show_doc(SchedNo)

sched = SchedNo(0, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0., 0., 0., 0.])

show_doc(SchedExp)

sched = SchedExp(1, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [1., 1.18921, 1.41421, 1.68179, 2.])

show_doc(SchedPoly)

sched = SchedPoly(0, 2, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.125, 0.5, 1.125, 2.])

# +
p = torch.linspace(0.,1,100)

pows = [0.5,1.,2.]
for e in pows:
    f = SchedPoly(2, 0, e)
    plt.plot(p, [f(o) for o in p], label=f'power {e}')
plt.legend();


# -

#|export
def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.
    pcts = tensor([0] + L(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    pct_lim = len(pcts) - 2
    def _inner(pos):
        idx = min((pos >= pcts).nonzero().max(), pct_lim)
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos.item())
    return _inner


# `pcts` must be a list of positive numbers that add up to 1 and is the same length as `scheds`. The generated function will use `scheds[0]` from 0 to `pcts[0]` then `scheds[1]` from `pcts[0]` to `pcts[0]+pcts[1]` and so forth.

p = torch.linspace(0.,1,100)
f = combine_scheds([0.3,0.7], [SchedCos(0.3,0.6), SchedCos(0.6,0.2)])
plt.plot(p, [f(o) for o in p]);

p = torch.linspace(0.,1,100)
f = combine_scheds([0.3,0.2,0.5], [SchedLin(0.,1.), SchedNo(1.,1.), SchedCos(1., 0.)])
plt.plot(p, [f(o) for o in p]);

#|hide
test_close([f(0.), f(0.15), f(0.3), f(0.4), f(0.5), f(0.7), f(1.)],
           [0., 0.5, 1., 1., 1., 0.65451, 0.])


#|export
def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds([pct,1-pct], [SchedCos(start, middle), SchedCos(middle, end)])


# This is a useful helper function for the [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html). `pct` is used for the `start` to `middle` part, `1-pct` for the `middle` to `end`. Handles floats or collection of floats. For example:

f = combined_cos(0.25,0.5,1.,0.)
plt.plot(p, [f(o) for o in p]);

#|hide
test_close([f(0.), f(0.1), f(0.25), f(0.5), f(1.)], [0.5, 0.67275, 1., 0.75, 0.])
f = combined_cos(0.25, np.array([0.25,0.5]), np.array([0.5,1.]), np.array([0.,0.]))
for a,b in zip([f(0.), f(0.1), f(0.25), f(0.5), f(1.)],
               [[0.25,0.5], [0.33638,0.67275], [0.5,1.], [0.375,0.75], [0.,0.]]):
    test_close(a,b)


# ## ParamScheduler -

#|export
@docs
class ParamScheduler(Callback):
    "Schedule hyper-parameters according to `scheds`"
    order,run_valid = 60,False

    def __init__(self, scheds): self.scheds = scheds
    def before_fit(self): self.hps = {p:[] for p in self.scheds.keys()}
    def before_batch(self): self._update_val(self.pct_train)

    def _update_val(self, pct):
        for n,f in self.scheds.items(): self.opt.set_hyper(n, f(pct))

    def after_batch(self):
        for p in self.scheds.keys(): self.hps[p].append(self.opt.hypers[-1][p])

    def after_fit(self):
        if hasattr(self.learn, 'recorder') and hasattr(self, 'hps'): self.recorder.hps = self.hps

    _docs = {"before_fit": "Initialize container for hyper-parameters",
             "before_batch": "Set the proper hyper-parameters in the optimizer",
             "after_batch": "Record hyper-parameters of this batch",
             "after_fit": "Save the hyper-parameters in the recorder if there is one"}


# `scheds` is a dictionary with one key for each hyper-parameter you want to schedule, with either a scheduler or a list of schedulers as values (in the second case, the list must have the same length as the the number of parameters groups of the optimizer).

learn = synth_learner()
sched = {'lr': SchedLin(1e-3, 1e-2)}
learn.fit(1, cbs=ParamScheduler(sched))
n = len(learn.dls.train)
test_close(learn.recorder.hps['lr'], [1e-3 + (1e-2-1e-3) * i/n for i in range(n)])


#|hide
#test discriminative lrs
def _splitter(m): return [[m.a], [m.b]]
learn = synth_learner(splitter=_splitter)
sched = {'lr': combined_cos(0.5, np.array([1e-4,1e-3]), np.array([1e-3,1e-2]), np.array([1e-5,1e-4]))}
learn.fit(1, cbs=ParamScheduler(sched))

show_doc(ParamScheduler.before_fit)

show_doc(ParamScheduler.before_batch)

show_doc(ParamScheduler.after_batch)

show_doc(ParamScheduler.after_fit)


#|export
@patch
def fit_one_cycle(self:Learner, n_epoch, lr_max=None, div=25., div_final=1e5, pct_start=0.25, wd=None,
                  moms=None, cbs=None, reset_opt=False, start_epoch=0):
    "Fit `self.model` for `n_epoch` using the 1cycle policy."
    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
              'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)


# The 1cycle policy was introduced by Leslie N. Smith et al. in [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120). It schedules the learning rate with a cosine annealing from `lr_max/div` to `lr_max` then `lr_max/div_final` (pass an array to `lr_max` if you want to use differential learning rates) and the momentum with cosine annealing according to the values in `moms`. The first phase takes `pct_start` of the training. You can optionally pass additional `cbs` and `reset_opt`.

#Integration test: training a few epochs should make the model better
learn = synth_learner(lr=1e-2)
xb,yb = learn.dls.one_batch()
init_loss = learn.loss_func(learn.model(xb), yb)
learn.fit_one_cycle(2)
xb,yb = learn.dls.one_batch()
final_loss = learn.loss_func(learn.model(xb), yb)
assert final_loss < init_loss

#Scheduler test
lrs,moms = learn.recorder.hps['lr'],learn.recorder.hps['mom']
test_close(lrs,  [combined_cos(0.25,1e-2/25,1e-2,1e-7)(i/20) for i in range(20)])
test_close(moms, [combined_cos(0.25,0.95,0.85,0.95)(i/20) for i in range(20)])


#|export
@patch
def plot_sched(self:Recorder, keys=None, figsize=None):
    keys = self.hps.keys() if keys is None else L(keys)
    rows,cols = (len(keys)+1)//2, min(2, len(keys))
    figsize = figsize or (6*cols,4*rows)
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if len(keys) > 1 else L(axs)
    for p,ax in zip(keys, axs):
        ax.plot(self.hps[p])
        ax.set_ylabel(p)


#|hide
#test discriminative lrs
def _splitter(m): return [[m.a], [m.b]]
learn = synth_learner(splitter=_splitter)
learn.fit_one_cycle(1, lr_max=slice(1e-3,1e-2))
#n = len(learn.dls.train)
#est_close(learn.recorder.hps['lr'], [1e-3 + (1e-2-1e-3) * i/n for i in range(n)])

learn = synth_learner()
learn.fit_one_cycle(2)

learn.recorder.plot_sched()


#|export
@patch
def fit_flat_cos(self:Learner, n_epoch, lr=None, div_final=1e5, pct_start=0.75, wd=None,
                 cbs=None, reset_opt=False, start_epoch=0):
    "Fit `self.model` for `n_epoch` at flat `lr` before a cosine annealing."
    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)
    lr = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr, lr, lr/div_final)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=0)


learn = synth_learner()
learn.fit_flat_cos(2)

learn.recorder.plot_sched()


#|export
@patch
def fit_sgdr(self:Learner, n_cycles, cycle_len, lr_max=None, cycle_mult=2, cbs=None, reset_opt=False, wd=None,
             start_epoch=0):
    "Fit `self.model` for `n_cycles` of `cycle_len` using SGDR."
    if self.opt is None: self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    n_epoch = cycle_len * (cycle_mult**n_cycles-1)//(cycle_mult-1)
    pcts = [cycle_len * cycle_mult**i / n_epoch for i in range(n_cycles)]
    scheds = [SchedCos(lr_max, 0) for _ in range(n_cycles)]
    scheds = {'lr': combine_scheds(pcts, scheds)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)


# This schedule was introduced by Ilya Loshchilov et al. in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). It consists of `n_cycles` that are cosine annealings from `lr_max` (defaults to the `Learner` lr) to 0, with a length of `cycle_len * cycle_mult**i` for the `i`-th cycle (first one is `cycle_len`-long, then we multiply the length by `cycle_mult` at each epoch). You can optionally pass additional `cbs` and `reset_opt`.

# +
#|slow
learn = synth_learner()
with learn.no_logging(): learn.fit_sgdr(3, 1)
test_eq(learn.n_epoch, 7)
iters = [k * len(learn.dls.train) for k in [0,1,3,7]]
for i in range(3):
    n = iters[i+1]-iters[i]
    #The start of a cycle can be mixed with the 0 of the previous cycle with rounding errors, so we test at +1
    test_close(learn.recorder.lrs[iters[i]+1:iters[i+1]], [SchedCos(learn.lr, 0)(k/n) for k in range(1,n)])

learn.recorder.plot_sched()


# -

#|export
@patch
@delegates(Learner.fit_one_cycle)
def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
              pct_start=0.3, div=5.0, **kwargs):
    "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
    self.freeze()
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)


learn.fine_tune(1)


# ## Resume training from checkpoint

#|hide
class InterruptCallback(Callback):
    def __init__(self, epoch):
        self._interupt_before = epoch
    def before_epoch(self):
        if self.epoch == self._interupt_before:
            raise CancelFitException


# To enable resuming from checkpoint make sure to save model and optimizer state. This can be done using [SaveModelCallback](https://docs.fast.ai/callback.tracker.html#SaveModelCallback.html) setting `(with_opt=True)`. If training is interrupted define `learn` using the same parameters as before, load model from checkpoint and pass `start_epoch` to `fit` call. The training will be resumed from `start_epoch` with properly scheduled  `lr`.

#|slow
with tempfile.TemporaryDirectory() as d:
    learn1 = synth_learner(path=d, cbs=SaveModelCallback(with_opt=True, fname="ckpt"))
    learn1.fit_one_cycle(5, cbs=InterruptCallback(2))
    
    learn2 = synth_learner(path=d)
    learn2 = learn2.load("ckpt")
    learn2.fit_one_cycle(5, start_epoch=2)
    
    fig, axs = plt.subplots(1,2, sharey=True)
    axs[0].plot(learn1.recorder.lrs)
    axs[1].plot(learn2.recorder.lrs)


# ## LRFind -

#|export
@docs
class LRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"
    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        if num_it < 6: num_it = 6
        self.scheds = {'lr': [SchedExp(s, e) for (s,e) in zip(start_lr,end_lr)
                             ] if is_listy(start_lr) else SchedExp(start_lr, end_lr)}
        self.num_it,self.stop_div = num_it,stop_div

    def before_fit(self):
        super().before_fit()
        path = self.path/self.model_dir
        path.mkdir(parents=True, exist_ok=True)
        self.tmp_d = tempfile.TemporaryDirectory(dir=path)
        self.tmp_p = Path(self.tmp_d.name).stem
        self.learn.save(f'{self.tmp_p}/_tmp')
        self.best_loss = float('inf')

    def before_batch(self): self._update_val(self.train_iter/self.num_it)

    def after_batch(self):
        super().after_batch()
        if self.smooth_loss < self.best_loss: self.best_loss = self.smooth_loss
        if self.smooth_loss > 4*self.best_loss and self.stop_div: raise CancelFitException()
        if self.train_iter >= self.num_it: raise CancelFitException()

    def before_validate(self): raise CancelValidException()

    def after_fit(self):
        self.learn.opt.zero_grad() # Needed before detaching the optimizer for future fits
        tmp_f = self.path/self.model_dir/self.tmp_p/'_tmp.pth'
        if tmp_f.exists():
            self.learn.load(f'{self.tmp_p}/_tmp', with_opt=True)
            self.tmp_d.cleanup()

    _docs = {"before_fit": "Initialize container for hyper-parameters and save the model",
             "before_batch": "Set the proper hyper-parameters in the optimizer",
             "after_batch": "Record hyper-parameters of this batch and potentially stop training",
             "after_fit": "Save the hyper-parameters in the recorder if there is one and load the original model",
             "before_validate": "Skip the validation part of training"}


#|cuda
from fastai.vision.all import *

# +
#|cuda
set_seed(99, True)
path = untar_data(URLs.PETS)/'images'

image_files = get_image_files(path)
if sys.platform == "win32" and IN_NOTEBOOK:
    image_files = random.choices(image_files, k=int(len(image_files)/8))
    print("Randomly select 1/8 files in NOTEBOOK on Windows to save time")

# pickle can't serializer lamda function.
def _label_func(x):
    return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, image_files, valid_pct=0.2,
    label_func=_label_func, item_tfms=Resize(224))

learn = vision_learner(dls, resnet18)
learn.fit(1)
learn.opt.state_dict()['state'][1]['grad_avg']
# -

#|slow
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=Path(d))
    init_a,init_b = learn.model.a,learn.model.b
    with learn.no_logging(): learn.fit(20, cbs=LRFinder(num_it=100))
    assert len(learn.recorder.lrs) <= 100
    test_eq(len(learn.recorder.lrs), len(learn.recorder.losses))
    #Check stop if diverge
    if len(learn.recorder.lrs) < 100: assert learn.recorder.losses[-1] > 4 * min(learn.recorder.losses)
    #Test schedule
    test_eq(learn.recorder.lrs, [SchedExp(1e-7, 10)(i/100) for i in range_of(learn.recorder.lrs)])
    #No validation data
    test_eq([len(v) for v in learn.recorder.values], [1 for _ in range_of(learn.recorder.values)])
    #Model loaded back properly
    test_eq(learn.model.a, init_a)
    test_eq(learn.model.b, init_b)
    test_eq(learn.opt.state_dict()['state'], [{}, {}])

show_doc(LRFinder.before_fit)

show_doc(LRFinder.before_batch)

show_doc(LRFinder.after_batch)

show_doc(LRFinder.before_validate)

# ### Suggestion Methods

# There are a few methodologies for suggesting a learning rate automatically and these as we will see can further be passed into `lr_find`. Currently four methods are supported, however to write your own it should look like a function that can accept `LRFinder`'s returned `lrs`, `losses`, as well as the `num_it`. 
# Your function should return an `x,y` coordinate that can be plotted, such as below:
#
#
# ```python
# def myfunc(lrs:list, losses:list, num_it:int) -> tuple(float, tuple(float,int)):
#     ...
#     return suggestion, (suggestion,loss_idx)
# ```
#
# If there are any more parameters to be passed in, you should pass in your `func` as a partial and specify them yourself, such as:
#
# ```python
# def myfunc(lrs:list, losses:list, num_it:int, pct_reduction:float) -> tuple(float, tuple(float,int)):
#     ...
#     return suggestion, (suggestion,loss_idx)
# ```
#
# ```python
# f = partial(myfunc, pct_reduction=.2)
# ```

# +
#|hide
learn = synth_learner()
with learn.no_logging(): learn.fit(20, cbs=LRFinder(num_it=100))

lrs,losses = tensor(learn.recorder.lrs[100//10:-5]),tensor(learn.recorder.losses[100//10:-5])


# -

#|export
def valley(lrs:list, losses:list, num_it:int):
    "Suggests a learning rate from the longest valley and returns its index"
    n = len(losses)
    max_start, max_end = 0,0

    # find the longest valley
    lds = [1]*n
    for i in range(1,n):
        for j in range(0,i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]
    
    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return float(lrs[idx]), (float(lrs[idx]), losses[idx])


# The `valley` algorithm was developed by [ESRI](https://forums.fast.ai/t/automated-learning-rate-suggester/44199/30) and takes the steepest slope roughly 2/3 through the longest valley in the LR plot, and is also the default for `Learner.lr_find`

#|hide
valley(lrs, losses, 100)


#|export
def slide(lrs:list, losses:list, num_it:int, lr_diff:int=15, thresh:float=.005, adjust_value:float=1.):
    "Suggests a learning rate following an interval slide rule and returns its index"
    losses = to_np(losses)
    loss_grad = np.gradient(losses)

    r_idx = -1
    l_idx = r_idx - lr_diff
    local_min_lr = lrs[l_idx]
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > thresh):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1
    
    suggestion = float(local_min_lr) * adjust_value
    idx = np.interp(np.log10(suggestion), np.log10(lrs), losses)
    return suggestion, (suggestion, idx)


# The `slide` rule is an algorithm developed by Andrew Chang out of Novetta, and is detailed [here](https://forums.fast.ai/t/automated-learning-rate-suggester/44199?u=muellerzr).

#|hide
slide(lrs, losses, 100)


#|export
def minimum(lrs:list, losses:list, num_it:int):
    "Suggests a learning rate one-tenth the minumum before divergance and returns its index"
    lr_min = lrs[losses.argmin()].item()
    loss_idx = losses[min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_min))]
    return lr_min/10, (lr_min, loss_idx)


#|hide
minimum(lrs, losses, 100)


#|export
def steep(lrs:list, losses:list, num_it:int) -> (float, tuple):
    "Suggests a learning rate when the slope is the steepest and returns its index"
    grads = (losses[1:]-losses[:-1]) / (lrs[1:].log()-lrs[:-1].log())
    lr_steep = lrs[grads.argmin()].item()
    loss_idx = losses[min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_steep))]
    return lr_steep, (lr_steep, loss_idx)


#|hide
steep(lrs, losses, 100)


#|export
@patch
def plot_lr_find(self:Recorder, skip_end=5, return_fig=True, suggestions=None, nms=None, **kwargs):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs    = self.lrs    if skip_end==0 else self.lrs   [:-skip_end]
    losses = self.losses if skip_end==0 else self.losses[:-skip_end]
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    if suggestions:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
        for (val, idx), nm, color in zip(suggestions, nms, colors):
            ax.plot(val, idx, 'o', label=nm, c=color)
        ax.legend(loc='best')


#|export
mk_class("SuggestionMethod", **{o.__name__.capitalize():o for o in [valley,slide,minimum,steep]},
         doc="All possible suggestion methods as convience attributes to get tab-completion and typo-proofing")


#|export
@patch
def lr_find(self:Learner, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, show_plot=True, suggest_funcs=(SuggestionMethod.Valley)):
    "Launch a mock training to find a good learning rate and return suggestions based on `suggest_funcs` as a named tuple"
    n_epoch = num_it//len(self.dls.train) + 1
    cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    with self.no_logging(): self.fit(n_epoch, cbs=cb)
    if suggest_funcs is not None:
        lrs, losses = tensor(self.recorder.lrs[num_it//10:-5]), tensor(self.recorder.losses[num_it//10:-5])
        nan_idxs = torch.nonzero(torch.isnan(losses.view(-1)))
        if len(nan_idxs) > 0:
            drop_idx = min(nan_idxs)
            lrs = lrs[:drop_idx]
            losses = losses[:drop_idx]
        _suggestions, nms = [], []
        for func in tuplify(suggest_funcs):
            nms.append(func.__name__ if not isinstance(func, partial) else func.func.__name__) # deal with partials
            _suggestions.append(func(lrs, losses, num_it))
        
        SuggestedLRs = collections.namedtuple('SuggestedLRs', nms)
        lrs, pnts = [], []
        for lr, pnt in _suggestions:
            lrs.append(lr)
            pnts.append(pnt)
        if show_plot: self.recorder.plot_lr_find(suggestions=pnts, nms=nms)
        return SuggestedLRs(*lrs)

    elif show_plot: self.recorder.plot_lr_find()


# First introduced by Leslie N. Smith in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf), the LR Finder trains the model with exponentially growing learning rates from `start_lr` to `end_lr` for `num_it` and stops in case of divergence (unless `stop_div=False`) then plots the losses vs the learning rates with a log scale. 
#
# A variety of learning rate suggestion algorithms can be passed into the function, by default we use the `valley` paradigm.

#|slow
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=Path(d))
    weights_pre_lr_find = L(learn.model.parameters())
    lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    weights_post_lr_find = L(learn.model.parameters())
test_eq(weights_pre_lr_find, weights_post_lr_find)
print(f"Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}\nSlide interval:\t{lr_slide:.2e}")

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


