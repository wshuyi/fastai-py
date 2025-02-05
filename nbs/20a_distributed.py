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
#|default_exp distributed
# -

#|export
from __future__ import annotations
from fastai.basics import *
from fastai.callback.progress import ProgressCallback
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fastai.data.load import _FakeLoader,_loaders
from fastai.optimizer import OptimWrapper
try: from accelerate import Accelerator
except ModuleNotFoundError: pass


# # Distributed training
#
# > Callbacks and helper functions to train in parallel or use distributed training

# When using multiple GPUs, you will most probably want to fit using distributed training. 
#
# Example use can be found:
#
# - In the form of a script with [examples/distrib.py](https://github.com/fastai/fastai/blob/master/nbs/examples/distrib.py)
# - Across all the App Examples with the [Notebook Launcher](https://docs.fast.ai/distributed_app_examples.html)
# - At the bottom of this notebook for more examples with `notebook_launcher`.
#
# To use distributed training, there are only three required steps:
#
# 1. Add `with learn.distrib_ctx():` before your `learn.fit` call
# 2. Either config Accelerate yourself by running `accelerate config` from the command line, or run:
# ```python
# from accelerate.utils import write_basic_config
# write_basic_config()
# ```
# 3. Run your training script with `accelerate launch scriptname.py ...args...`
#
#
# If you're using `untar_data`, or may be downloading or uncompressing data or models as part of your script, you should wrap that code with `rank0_first`, which forces that step to occur first just once on the master process, prior to the remaining processes running it in parallel. E.g. instead of:
#
# ```python
# path = untar_data(URLs.IMAGEWOOF_320)
# ```
#
# ...you instead use:
#
# ```python
# path = rank0_first(untar_data, URLs.IMAGEWOOF_320)
# ```
#
# See below for details on the full API and underlying helper functions, if needed -- however, note that you will not need anything except the above unless you need to change how the distributed training is implemented.

# ## Parallel

#|export
@patch
def reset(self: DataParallel):
    "Patch required `reset` call into `DataParallel`"
    if hasattr(self.module, 'reset'): self.module.reset()


#|export
class ParallelTrainer(Callback):
    "Wrap a model `DataParallel` automatically"
    run_after,run_before = TrainEvalCallback,Recorder
    def __init__(self, device_ids): self.device_ids = device_ids
    def before_fit(self): self.learn.model = DataParallel(self.learn.model, device_ids=self.device_ids)
    def after_fit(self): self.learn.model = self.learn.model.module


#|export
@patch
def to_parallel(self: Learner, device_ids=None):
    "Add `ParallelTrainer` callback to a `Learner`"
    self.add_cb(ParallelTrainer(device_ids))
    return self


#|export
@patch
def detach_parallel(self: Learner):
    "Remove `ParallelTrainer` callback from a Learner"
    self.remove_cb(ParallelTrainer)
    return self


#|export
@patch
@contextmanager
def parallel_ctx(self: Learner, device_ids=None):
    "A context manager to adapt a learner to train in data parallel mode."
    try:
        self.to_parallel(device_ids)
        yield self
    finally: self.detach_parallel()


# ## Distributed

# ### Helper functions

#|export
@patch
def reset(self: DistributedDataParallel):
    "Patch required `reset` call into `DistributedDataParallel`"
    if hasattr(self.module, 'reset'): self.module.reset()


#|export
def setup_distrib(gpu=None):
    "Setup this process to participate in distributed training"
    if gpu is None: return gpu
    gpu = int(gpu)
    torch.cuda.set_device(int(gpu))
    if num_distrib() > 0: torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu


#|export
def teardown_distrib():
    "Free distributed training resources"
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


# ### DataLoader

#|export
def _round_to_multiple(number,multiple): return int(math.ceil(number/multiple)*multiple)


#|export
class DistributedDL(TfmdDL):
    "A `TfmdDL` which splits a batch into equal size pieces for each worker"
    def __init__(self,dl,rank=None,world_size=None,device=None):
        if rank is None: rank=rank_distrib()
        if world_size is None: world_size=num_distrib()
        store_attr()
        if type(dl) == torch.utils.data.DataLoader:
            shuffle = True if eq(type(dl.sampler), torch.utils.data.RandomSampler) else False
            self.dl = DataLoader(dataset=dl.dataset, bs=dl.batch_size, num_workers=dl.num_workers, \
                pin_memory=dl.pin_memory, timeout=dl.timeout, shuffle=shuffle, drop_last=dl.drop_last, persistent_workers=dl.persistent_workers)
        self.bs,self.drop_last,self.dataset,fake,self.num_workers,self.offs,self.pin_memory = \
            attrgetter('bs','drop_last','dataset','fake_l','num_workers','offs','pin_memory')(self.dl)
        if device is None: self.device = self.dl.device
        self.fake_l = _FakeLoader(self, fake.pin_memory, fake.num_workers, fake.timeout, 
                                  persistent_workers=fake.persistent_workers, 
                                  pin_memory_device=fake.pin_memory_device)
        
    def _broadcast(self,t,rank):
        "Broadcasts t from rank `rank` to all other ranks. Returns t so t is same for all ranks after call."
        t = LongTensor(t).cuda() # nccl only works with cuda tensors
        torch.distributed.broadcast(t,rank)
        return t.cpu().tolist()

    def _to_detach(self,b,cpu=True,gather=True): return to_detach(b,cpu,gather) # member func so we can override for test
    def __len__(self): return _round_to_multiple(len(self.dl),self.world_size)//self.world_size
    def get_idxs(self):
        idxs = list(self.dl.get_idxs()) # compute get_idxs in all ranks (we'll only use rank 0 but size must be consistent)
        idxs = self._broadcast(idxs,0)  # broadcast and receive it from rank 0 to all
        self.n = len(idxs)              # we assumed n was dl.n but we really care about number of idxs
        # add extra samples to make it evenly divisible
        self.n_padded = _round_to_multiple(self.n,self.world_size)
        idxs += (idxs * (self.n_padded//self.n))[:self.n_padded-self.n] # idx needs to be repeated when n_padded>>n
        # slice padded idxs so that each rank gets self.n_padded//self.world_size tensors
        return idxs[self.rank*self.n_padded//self.world_size:(self.rank+1)*self.n_padded//self.world_size]

    def before_iter(self):
        self.i = 0
        self.dl.before_iter()

    def randomize(self): self.dl.randomize()
    def after_batch(self,b):
        self.i += find_bs(b)
        return self.dl.after_batch(b)

    def after_iter(self):  self.dl.after_iter()
    def create_batches(self,samps): return self.dl.create_batches(samps)
    def to_detach(self,b, cpu=True, gather=True):
        b = self._to_detach(b, cpu, gather)
        def _inner(b):
            if b.ndim>0:
                # for each rank, compute overflow of read idxs vs self.n and accumulate them to unpad totals after gathering
                n = sum([min(0,max(-len(b)//self.world_size,
                                   self.n-(self.i+r*self.n_padded//self.world_size))) for r in range(self.world_size)])
                b = b[:n or None]
            return b
        return apply(_inner,b) if gather and all(hasattr(self,o) for o in ('i','n','n_padded')) else b


#|hide
_tmp_file = tempfile.NamedTemporaryFile().name
# patch _broadcast with a mocked version so we can test DistributedDL w/o a proper DDP setup
@patch
def _broadcast(self:DistributedDL,t,rank):
    t = LongTensor(t)
    if rank == self.rank: torch.save(t,_tmp_file)
    else:                 t.data = torch.load(_tmp_file)
    return t.tolist()
# patch _to_detach with a mocked version that will return right gathered size but -100 for other rank tensors
@patch
def _to_detach(self:DistributedDL,b,cpu=True,gather=True):
    b = to_detach(b,cpu,gather)
    if not gather: return b
    def _inner(b, cpu, gather):
        if b.ndim == 0: b=b[None]
        b = torch.cat([b if i==self.rank else torch.full_like(b,-100) for i in range(self.world_size)])
        return b if b.ndim > 0 else b.mean()
    return apply(_inner,b,cpu,gather)


dl = TfmdDL(list(range(50)), bs=12, num_workers=2)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), (torch.arange(i*13, i*13+12)%50,torch.tensor([i*13+12])%50))

#|hide
dl = torch.utils.data.DataLoader(list(range(50)), batch_size=12, num_workers=2)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), (torch.arange(i*13, i*13+12)%50,torch.tensor([i*13+12])%50))

#|hide
dl = TfmdDL(list(zip(range(50),range(100,150))), bs=12, num_workers=4)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), [(torch.arange(i*13, i*13+12)%50,100+torch.arange(i*13, i*13+12)%50),
                        ((torch.tensor([i*13+12])%50),100+torch.tensor([i*13+12])%50)])

#|hide
dl = TfmdDL(list(range(50)), bs=12, num_workers=2,drop_last=True)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), [torch.arange(i*13, i*13+12)%50])

#|hide
dl = TfmdDL(list(zip(range(12),range(100,112))), bs=12, num_workers=4)
res,dls = [],[]
for i in range(5): dls.append(DistributedDL(dl, i, 5))
for b in zip(*dls):
    for r in range(5):
        d=L(dls[r].to_detach(b[r]))
        test_eq(d.map(lambda x:(x!=-100).sum().item()),(3,3) if r!=4 else (0,0))

#|hide
dl = TfmdDL(list(range(10)), bs=4, num_workers=2, shuffle=True)
res = []
for i in range(3):
    dl1 = DistributedDL(dl, i, 3)
    b  = list(dl1)[0]
    bd = dl1.to_detach(b)
    test_eq(b[:None if i<2 else 2],bd[4*i:4*(i+1)])

#|hide
from fastai.callback.data import WeightedDL

#|hide
dl = WeightedDL(list(range(50)), bs=16, num_workers=2, shuffle=True,wgts=list(np.arange(50)>=25))
res = []
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    res += list(dl1)[0].tolist()
test(res,[25]*len(res),operator.ge)        # all res >=25
test(res,[25]*len(res),lambda a,b: ~(a<b)) # all res NOT < 25

# ### DistributedTrainer -

#|export
_hidden_params = ["mixed_precision", "fp16", "log_with", "logging_dir", "step_scheduler_with_optimizer"]


#|export
class DistributedTrainer(Callback):
    "Wrap `model` in `DistributedDataParallel` and `dls` in `DistributedDL`"
    order = 11
    @delegates(Accelerator, but=_hidden_params)
    def __init__(self,
        sync_bn=True, # Whether to replace all batch norm with `nn.SyncBatchNorm`
        **kwargs
    ):
        store_attr()
        self.accelerator = Accelerator(**kwargs)
    def before_fit(self):
        self.learn.model = self.accelerator.prepare(
            nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model
        )
        self.old_dls = list(self.dls)
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
        if rank_distrib(): self.learn.logger=noop

    def _wrap_dl(self, dl): return dl if isinstance(dl,DistributedDL) else DistributedDL(dl, device=self.learn.model.device)
    def _backward(self): self.accelerator.backward(self.learn.loss_grad)
    
    def before_train(self):    self.learn.dl = self._wrap_dl(self.learn.dl)
    def before_validate(self): self.learn.dl = self._wrap_dl(self.learn.dl)
    def after_fit(self): self.learn.model,self.learn.dls.loaders = self.learn.model.module,self.old_dls


#|export
@patch
@delegates(Accelerator, but=_hidden_params)
def to_distributed(self: Learner,
        sync_bn=True, # Whether to replace all batch norm with `nn.SyncBatchNorm`
        **kwargs
    ):
    "Add `AcceleratedTrainer` to a learner, and configures an Accelerator"
    self.add_cb(DistributedTrainer(sync_bn, **kwargs))
    if rank_distrib(): self.remove_cb(ProgressCallback)
    return self


#|export
@patch
def detach_distributed(self: Learner):
    "Remove `DistributedTrainer` from a learner"
    if num_distrib() <=1: return self
    self.remove_cb(DistributedTrainer)
    if rank_distrib() and not hasattr(self, 'progress'): self.add_cb(ProgressCallback())
    return self


# ### `distrib_ctx` context manager

#|export
@patch
@contextmanager
@delegates(Accelerator, but=_hidden_params)
def distrib_ctx(self: Learner,
        sync_bn=True, # Whether to replace all batch norm with `nn.SyncBatchNorm`
        in_notebook=False, # Whether we are launching from a notebook or not
        **kwargs
   ):
    "A context manager to adapt a learner to train in distributed data parallel mode."
    try: import accelerate
    except ImportError as e: 
        e.args = ["Accelerate is required. Install with `pip install accelerate`"]
        raise
    # Adapt self to DistributedDataParallel, yield, and cleanup afterwards.
    cleanup_dpg = False
    try:
        if in_notebook:
            cuda_id = rank_distrib()
            if not torch.distributed.is_initialized():
                setup_distrib(cuda_id)
                cleanup_dpg = torch.distributed.is_initialized()
            if not rank_distrib(): print("Training Learner...")
        if num_distrib(): self.to_distributed(sync_bn, **kwargs)
        yield self
    finally:
        self.detach_distributed()
        if cleanup_dpg: teardown_distrib()


# `distrib_ctx` prepares a learner to train in distributed data parallel mode. It assumes the script/code will either be ran through the command line via `accelerate launch` or through the `notebook_launcher` function from Accelerate. It also assumes that `accelerate` has been configured through either running `write_basic_config()` or calling `accelerate config` through the CLI and answering the prompts.
#
# Typical usage:
#
# ```
# with learn.distrib_ctx(): learn.fit(.....)
# ```
#
# It attaches a `DistributedTrainer` callback and `DistributedDL` data loader to  the learner, then executes `learn.fit(.....)`.  Upon exiting the context, it removes the `DistributedTrainer` and `DistributedDL`, and destroys any locally created distributed process group.  The process is still attached to the GPU though.

#|export
def rank0_first(func, *args, **kwargs):
    "Execute `func` in the Rank-0 process first, then in other ranks in parallel."
    if args or kwargs: func = partial(func, *args, **kwargs)
    dummy_l = Learner(DataLoaders(device='cpu'), nn.Linear(1,1), loss_func=lambda: 0)
    with dummy_l.distrib_ctx():
        if not rank_distrib(): res = func()
        distrib_barrier()
        if rank_distrib(): res = func()
    return res


# `rank0_first` calls `f()` in rank-0 process first, then in parallel on the rest, in distributed training mode. In single process, non-distributed training mode, `f()` is called only once as expected.
#
# One application of `rank0_first()` is to make fresh downloads via `untar_data` safe in distributed training scripts launched by `python -m fastai.launch <script>`:
#
# <code>path = untar_data(URLs.IMDB)</code>
#
# becomes:
#
# <code>path = rank0_first(lambda: untar_data(URLs.IMDB))</code>
#
# Some learner factory methods may use `untar_data` to download pretrained models:
#
# <code>learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)</code>
#
# becomes:
#
# <code>learn = rank0_first(lambda: text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy))</code>
#
# Otherwise, multiple processes will download at the same time and corrupt the data.

# ## Notebook Launcher
#
# Accelerate provides a [notebook_launcher](https://huggingface.co/docs/accelerate/launcher) functionality to let you keep using your Jupyter Notebook as you would, but train in a distributed setup!
#
# First, make sure accelerate is properly configured. You can either run `accelerate config` from the command line, or have an autofilled configuration setup by running in the first cell of your notebook:
#
# ```python
# from accelerate.utils import write_basic_config
# write_basic_config()
# ```
# After Accelerate is configured, to utilize the `notebook_launcher` functionality migrate your training into a function, and pass this to `notebook_launcher`, such as:
#
# ```python
# ---
# from fastai.vision.all import *
# from fastai.distributed import *
#
# def train():
#     set_seed(99, True)
#     path = untar_data(URLs.PETS)/'images'
#     dls = ImageDataLoaders.from_name_func(
#         path, get_image_files(path), valid_pct=0.2,
#         label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
#     
#     learn = vision_learner(dls, resnet34, metrics=error_rate).to_fp16()
#     with learn.distrib_ctx(in_notebook=True):
#         learn.fine_tune(1)
# ---
# from accelerate import notebook_launcher
# notebook_launcher(train, num_processes=2)
# ---
# ```

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


