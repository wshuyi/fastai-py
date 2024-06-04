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

#|export
from __future__ import annotations
from fastai.torch_basics import *
from fastai.tabular.core import *

#|hide
from nbdev.showdoc import *


# +
#|default_exp tabular.model
# -

# # Tabular model
#
# > A basic model that can be used on tabular data

# ## Embeddings

#|export
def emb_sz_rule(
    n_cat:int # Cardinality of a category
) -> int:
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))


#|export
def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz


# Through trial and error, this general rule takes the lower of two values:
#
# * A dimension space of 600
# * A dimension space equal to 1.6 times the cardinality of the variable to 0.56.
#
# This provides a good starter for a good embedding space for your variables. For more advanced users who wish to lean into this practice, you can tweak these values to your discretion. It is not uncommon for slight adjustments to this general formula to provide more success.

#|export
def get_emb_sz(
    to:Tabular|TabularPandas, 
    sz_dict:dict=None # Dictionary of {'class_name' : size, ...} to override default `emb_sz_rule` 
) -> list: # List of embedding sizes for each category
    "Get embedding size for each cat_name in `Tabular` or `TabularPandas`, or populate embedding size manually using sz_dict"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]


#|export
class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, 
        emb_szs:list, # Sequence of (num_embeddings, embedding_dim) for each categorical variable
        n_cont:int, # Number of continuous variables
        out_sz:int, # Number of outputs for final `LinBnDrop` layer
        layers:list, # Sequence of ints used to specify the input and output size of each `LinBnDrop` layer
        ps:float|MutableSequence=None, # Sequence of dropout probabilities for `LinBnDrop`
        embed_p:float=0., # Dropout probability for `Embedding` layer
        y_range=None, # Low and high for `SigmoidRange` activation 
        use_bn:bool=True, # Use `BatchNorm1d` in `LinBnDrop` layers
        bn_final:bool=False, # Use `BatchNorm1d` on final layer
        bn_cont:bool=True, # Use `BatchNorm1d` on continuous variables
        act_cls=nn.ReLU(inplace=True), # Activation type for `LinBnDrop` layers
        lin_first:bool=True # Linear layer is first or last in `LinBnDrop` layers
    ):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)


# This model expects your `cat` and `cont` variables seperated. `cat` is passed through an `Embedding` layer and potential `Dropout`, while `cont` is passed though potential `BatchNorm1d`. Afterwards both are concatenated and passed through a series of `LinBnDrop`, before a final `Linear` layer corresponding to the expected outputs. 

emb_szs = [(4,2), (17,8)]
m = TabularModel(emb_szs, n_cont=2, out_sz=2, layers=[200,100]).eval()
x_cat = torch.tensor([[2,12]]).long()
x_cont = torch.tensor([[0.7633, -0.1887]]).float()
out = m(x_cat, x_cont)


#|export
@delegates(TabularModel.__init__)
def tabular_config(**kwargs):
    "Convenience function to easily create a config for `TabularModel`"
    return kwargs


# Any direct setup of `TabularModel`'s internals should be passed through here:

config = tabular_config(embed_p=0.6, use_bn=False); config

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()
