# ---
# skip_exec: true
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

# +
#| default_exp vision.widgets
# -

#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

#|export
from __future__ import annotations
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.vision.core import *
from fastcore.parallel import *
from ipywidgets import HBox,VBox,widgets,Button,Checkbox,Dropdown,Layout,Box,Output,Label,FileUpload

#|hide
from nbdev.showdoc import *

#|export
_all_ = ['HBox','VBox','widgets','Button','Checkbox','Dropdown','Layout','Box','Output','Label','FileUpload']


# # Vision widgets
#
# > ipywidgets for images

#|export
@patch
def __getitem__(self:Box, i): return self.children[i]


#|export
def widget(im, *args, **layout) -> Output:
    "Convert anything that can be `display`ed by `IPython` into a widget"
    o = Output(layout=merge(*args, layout))
    with o: display(im)
    return o


show_doc(widget)

im = Image.open('images/puppy.jpg').to_thumb(256,512)
VBox([widgets.HTML('Puppy'),
      widget(im, max_width="192px")])


#|export
def _update_children(
    change:dict # A dictionary holding the information about the changed widget
):
    "Sets a value to the `layout` attribute on widget initialization and change"
    for o in change['owner'].children:
        if not o.layout.flex: o.layout.flex = '0 0 auto'


#|export
def carousel(
    children:tuple|MutableSequence=(), # `Box` objects to display in a carousel
    **layout
) -> Box: # An `ipywidget`'s carousel
    "A horizontally scrolling carousel"
    def_layout = dict(overflow='scroll hidden', flex_flow='row', display='flex')
    res = Box([], layout=merge(def_layout, layout))
    res.observe(_update_children, names='children')
    res.children = children
    return res


show_doc(carousel)

# +
ts = [VBox([widget(im, max_width='192px'), Button(description='click')])
      for o in range(3)]

carousel(ts, width='450px')


# -

#|export
def _open_thumb(
    fn:Path|str, # A path of an image
    h:int, # Thumbnail Height
    w:int # Thumbnail Width
) -> Image: # `PIL` image to display
    "Opens an image path and returns the thumbnail of the image"
    return Image.open(fn).to_thumb(h, w).convert('RGBA')


#|export
class ImagesCleaner:
    "A widget that displays all images in `fns` along with a `Dropdown`"
    def __init__(self,
        opts:tuple=(), # Options for the `Dropdown` menu
        height:int=128, # Thumbnail Height
        width:int=256, # Thumbnail Width
        max_n:int=30 # Max number of images to display
    ):
        opts = ('<Keep>', '<Delete>')+tuple(opts)
        store_attr('opts,height,width,max_n')
        self.widget = carousel(width='100%')

    def set_fns(self,
        fns:list # Contains a path to each image 
    ):
        "Sets a `thumbnail` and a `Dropdown` menu for each `VBox`"
        self.fns = L(fns)[:self.max_n]
        ims = parallel(_open_thumb, self.fns, h=self.height, w=self.width, progress=False,
                       n_workers=min(len(self.fns)//10,defaults.cpus))
        self.widget.children = [VBox([widget(im, height=f'{self.height}px'), Dropdown(
            options=self.opts, layout={'width': 'max-content'})]) for im in ims]

    def _ipython_display_(self): display(self.widget)
    def values(self) -> list:
        "Current values of `Dropdown` for each `VBox`"
        return L(self.widget.children).itemgot(1).attrgot('value')
    def delete(self) -> list:
        "Indices of items to delete"
        return self.values().argwhere(eq('<Delete>'))
    def change(self) -> list:
        "Tuples of the form (index of item to change, new class)"
        idxs = self.values().argwhere(not_(in_(['<Delete>','<Keep>'])))
        return idxs.zipwith(self.values()[idxs])


show_doc(ImagesCleaner)

fns = get_image_files('images')
w = ImagesCleaner(('A','B'))
w.set_fns(fns)
w

w.delete(),w.change()


#|export
def _get_iw_info(
    learn,
    ds_idx:int=0 # Index in `learn.dls`
) -> list:
    "For every image in `dls` `zip` it's `Path`, target and loss"
    dl = learn.dls[ds_idx].new(shuffle=False, drop_last=False)
    probs,targs,preds,losses = learn.get_preds(dl=dl, with_input=False, with_loss=True, with_decoded=True)
    targs = [dl.vocab[t] for t in targs]
    return L([dl.dataset.items,targs,losses]).zip()


#|export
@delegates(ImagesCleaner)
class ImageClassifierCleaner(GetAttr):
    "A widget that provides an `ImagesCleaner` for a CNN `Learner`"
    def __init__(self, learn, **kwargs):
        vocab = learn.dls.vocab
        self.default = self.iw = ImagesCleaner(vocab, **kwargs)
        self.dd_cats = Dropdown(options=vocab)
        self.dd_ds   = Dropdown(options=('Train','Valid'))
        self.iwis = _get_iw_info(learn,0),_get_iw_info(learn,1)
        self.dd_ds.observe(self.on_change_ds, 'value')
        self.dd_cats.observe(self.on_change_ds, 'value')
        self.on_change_ds()
        self.widget = VBox([self.dd_cats, self.dd_ds, self.iw.widget])

    def _ipython_display_(self): display(self.widget)
    def on_change_ds(self,change=None):
        "Toggle between training validation set view"
        info = L(o for o in self.iwis[self.dd_ds.index] if o[1]==self.dd_cats.value)
        self.iw.set_fns(info.sorted(2, reverse=True).itemgot(0))


show_doc(ImageClassifierCleaner)

# # Export -

#|hide
import nbdev; nbdev.nbdev_export()


