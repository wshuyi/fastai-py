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

#|export
from __future__ import annotations
from fastai.imports import *
from fastai.data.all import *
from fastai.optimizer import *
from fastai.learner import *
from fastai.callback.core import *
from torch.utils.data import TensorDataset

# +
#|default_exp test_utils
# -

# # Synthetic Learner
#
# > For quick testing of the training loop and Callbacks

#|export
from torch.utils.data import TensorDataset


#|export
def synth_dbunch(a=2, b=3, bs=16, n_train=10, n_valid=2, cuda=False):
    def get_data(n):
        x = torch.randn(bs*n, 1)
        return TensorDataset(x, a*x + b + 0.1*torch.randn(bs*n, 1))
    train_ds = get_data(n_train)
    valid_ds = get_data(n_valid)
    device = default_device() if cuda else None
    train_dl = TfmdDL(train_ds, bs=bs, shuffle=True, num_workers=0)
    valid_dl = TfmdDL(valid_ds, bs=bs, num_workers=0)
    return DataLoaders(train_dl, valid_dl, device=device)


#|export
class RegModel(Module):
    def __init__(self): self.a,self.b = nn.Parameter(torch.randn(1)),nn.Parameter(torch.randn(1))
    def forward(self, x): return x*self.a + self.b


#|export
@delegates(Learner.__init__)
def synth_learner(n_trn=10, n_val=2, cuda=False, lr=1e-3, data=None, model=None, **kwargs):
    if data is None: data=synth_dbunch(n_train=n_trn,n_valid=n_val, cuda=cuda)
    if model is None: model=RegModel()
    return Learner(data, model, lr=lr, loss_func=MSELossFlat(),
                   opt_func=partial(SGD, mom=0.9), **kwargs)


#|export
class VerboseCallback(Callback):
    "Callback that prints the name of each event called"
    def __call__(self, event_name):
        print(event_name)
        super().__call__(event_name)


# ## Install Utils

#|export
def get_env(name):
    "Return env var value if it's defined and not an empty string, or return Unknown"
    res = os.environ.get(name,'')
    return res if len(res) else "Unknown"


#|export
def try_import(module):
    "Try to import `module`. Returns module's object on success, None on failure"
    try: return importlib.import_module(module)
    except: return None


#|export
def nvidia_smi(cmd = "nvidia-smi"):
    try: res = run(cmd)
    except OSError as e: return None
    return res


res = nvidia_smi()


#|export
def nvidia_mem():
    try: mem = run("nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader")
    except: return None
    return mem.strip().split('\n')


nvidia_mem()


#|export
def show_install(show_nvidia_smi:bool=False):
    "Print user's setup information"

    import fastai, platform, fastprogress, fastcore

    rep = []
    opt_mods = []

    rep.append(["=== Software ===", None])
    rep.append(["python", platform.python_version()])
    rep.append(["fastai", fastai.__version__])
    rep.append(["fastcore", fastcore.__version__])
    rep.append(["fastprogress", fastprogress.__version__])
    rep.append(["torch",  torch.__version__])

    # nvidia-smi
    smi = nvidia_smi()
    if smi:
        match = re.findall(r'Driver Version: +(\d+\.\d+)', smi)
        if match: rep.append(["nvidia driver", match[0]])

    available = "available" if torch.cuda.is_available() else "**Not available** "
    rep.append(["torch cuda", f"{torch.version.cuda} / is {available}"])

    # no point reporting on cudnn if cuda is not available, as it
    # seems to be enabled at times even on cpu-only setups
    if torch.cuda.is_available():
        enabled = "enabled" if torch.backends.cudnn.enabled else "**Not enabled** "
        rep.append(["torch cudnn", f"{torch.backends.cudnn.version()} / is {enabled}"])

    rep.append(["\n=== Hardware ===", None])

    gpu_total_mem = []
    nvidia_gpu_cnt = 0
    if smi:
        mem = nvidia_mem()
        nvidia_gpu_cnt = len(ifnone(mem, []))

    if nvidia_gpu_cnt: rep.append(["nvidia gpus", nvidia_gpu_cnt])

    torch_gpu_cnt = torch.cuda.device_count()
    if torch_gpu_cnt:
        rep.append(["torch devices", torch_gpu_cnt])
        # information for each gpu
        for i in range(torch_gpu_cnt):
            rep.append([f"  - gpu{i}", (f"{gpu_total_mem[i]}MB | " if gpu_total_mem else "") + torch.cuda.get_device_name(i)])
    else:
        if nvidia_gpu_cnt:
            rep.append([f"Have {nvidia_gpu_cnt} GPU(s), but torch can't use them (check nvidia driver)", None])
        else:
            rep.append([f"No GPUs available", None])


    rep.append(["\n=== Environment ===", None])

    rep.append(["platform", platform.platform()])

    if platform.system() == 'Linux':
        distro = try_import('distro')
        if distro:
            # full distro info
            rep.append(["distro", ' '.join(distro.linux_distribution())])
        else:
            opt_mods.append('distro');
            # partial distro info
            rep.append(["distro", platform.uname().version])

    rep.append(["conda env", get_env('CONDA_DEFAULT_ENV')])
    rep.append(["python", sys.executable])
    rep.append(["sys.path", "\n".join(sys.path)])

    print("\n\n```text")

    keylen = max([len(e[0]) for e in rep if e[1] is not None])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] is not None else ""))

    if smi:
        if show_nvidia_smi: print(f"\n{smi}")
    else:
        if torch_gpu_cnt: print("no nvidia-smi is found")
        else: print("no supported gpus found on this system")

    print("```\n")

    print("Please make sure to include opening/closing ``` when you paste into forums/github to make the reports appear formatted as code sections.\n")

    if opt_mods:
        print("Optional package(s) to enhance the diagnostics can be installed with:")
        print(f"pip install {' '.join(opt_mods)}")
        print("Once installed, re-run this utility to get the additional information")

#|hide
show_install(True)

# ## - Export

#|hide
from nbdev import *
nbdev_export()


