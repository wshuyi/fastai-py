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

# # Set up PyTorch links to appear in fastai docs
# - This is mainly for internal use

# +
#|default_exp _pytorch_doc
# -

from fastai.basics import *

# Test links

#|export
from __future__ import annotations
from types import ModuleType

#|export
PYTORCH_URL = 'https://pytorch.org/docs/stable/'


#|export
def _mod2page(
    mod:ModuleType, # A PyTorch module
) -> str:
    "Get the webpage name for a PyTorch module"
    if mod == Tensor: return 'tensors.html'
    name = mod.__name__
    name = name.replace('torch.', '').replace('utils.', '')
    if name.startswith('nn.modules'): return 'nn.html'
    return f'{name}.html'


test_eq(_mod2page(Tensor), 'tensors.html')
test_eq(_mod2page(torch.nn), 'nn.html')
test_eq(_mod2page(inspect.getmodule(nn.Conv2d)), 'nn.html')
test_eq(_mod2page(F), 'nn.functional.html')
test_eq(_mod2page(torch.optim), 'optim.html')
test_eq(_mod2page(torch.utils.data), 'data.html')

#|export
import importlib


#|export
def pytorch_doc_link(
    name:str # Name of a PyTorch module, class or function
) -> (str, None):
    "Get the URL to the documentation of a PyTorch module, class or function"
    if name.startswith('F'): name = 'torch.nn.functional' + name[1:]
    if not name.startswith('torch.'): name = 'torch.' + name
    if name == 'torch.Tensor': return f'{PYTORCH_URL}tensors.html'
    try:
        mod = importlib.import_module(name)
        return f'{PYTORCH_URL}{_mod2page(mod)}'
    except: pass
    splits = name.split('.')
    mod_name,fname = '.'.join(splits[:-1]),splits[-1]
    if mod_name == 'torch.Tensor': return f'{PYTORCH_URL}tensors.html#{name}'
    try:
        mod = importlib.import_module(mod_name)
        page = _mod2page(mod)
        return f'{PYTORCH_URL}{page}#{name}'
    except: return None


test_links = {
    'Tensor': 'https://pytorch.org/docs/stable/tensors.html',
    'Tensor.sqrt': 'https://pytorch.org/docs/stable/tensors.html#torch.Tensor.sqrt',
    'torch.zeros_like': 'https://pytorch.org/docs/stable/torch.html#torch.zeros_like',
    'nn.Module': 'https://pytorch.org/docs/stable/nn.html#torch.nn.Module',
    'nn.Linear': 'https://pytorch.org/docs/stable/nn.html#torch.nn.Linear',
    'F.cross_entropy': 'https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cross_entropy'
}
for f,l in test_links.items(): test_eq(pytorch_doc_link(f), l)

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


