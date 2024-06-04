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

# # Notebook Launcher examples
#
# A quick(ish) test of most of the main applications people use, taken from `fastbook`, and ran with Accelerate across multiple GPUs through `notebook_launcher`

# + active=""
# ---
# skip_exec: true
# ---

# +
from fastai.vision.all import *
from fastai.text.all import *
from fastai.tabular.all import *
from fastai.collab import *

from accelerate import notebook_launcher
from fastai.distributed import *
# -

# :::{.callout-important}
#
# Before running, ensure that Accelerate has been configured through either `accelerate config` in the command line or by running `write_basic_config`
#
# :::

# +
# from accelerate.utils import write_basic_config
# write_basic_config()
# -

# ### Image Classification

# +
path = untar_data(URLs.PETS)/'images'

def train():
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2,
        label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    learn = vision_learner(dls, resnet34, metrics=error_rate).to_fp16()
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fine_tune(1)

notebook_launcher(train, num_processes=2)
# -

# ### Image Segmentation

# +
path = untar_data(URLs.CAMVID_TINY)

def train():
    dls = SegmentationDataLoaders.from_label_func(
        path, bs=8, fnames = get_image_files(path/"images"),
        label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
        codes = np.loadtxt(path/'codes.txt', dtype=str)
    )
    learn = unet_learner(dls, resnet34)
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fine_tune(8)
        
notebook_launcher(train, num_processes=2)
# -

# ### Text Classification

# +
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')

def train():
    imdb_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72), CategoryBlock),
                      get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())
    dls = imdb_clas.dataloaders(df, bs=64)
    learn = rank0_first(lambda: text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy))
    with learn.distrib_ctx(in_notebook=True):
        learn.fine_tune(4, 1e-2)
        
notebook_launcher(train, num_processes=2)
# -

# ### Tabular

# +
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')


def train():
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
            cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race'],
            cont_names = ['age', 'fnlwgt', 'education-num'],
            procs = [Categorify, FillMissing, Normalize])

    learn = tabular_learner(dls, metrics=accuracy)
    with learn.distrib_ctx(in_notebook=True):
        learn.fit_one_cycle(3)
        
notebook_launcher(train, num_processes=2)
# -

# ### Collab Filtering

# +
path = untar_data(URLs.ML_SAMPLE)
df = pd.read_csv(path/'ratings.csv')

def train():
    dls = CollabDataLoaders.from_df(df)
    learn = collab_learner(dls, y_range=(0.5,5.5))
    with learn.distrib_ctx(in_notebook=True):
        learn.fine_tune(6)
        
notebook_launcher(train, num_processes=2)
# -

# ### Keypoints

# +
path = untar_data(URLs.BIWI_HEAD_POSE)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])

img_files = get_image_files(path)
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)


def train():
    biwi = DataBlock(
            blocks=(ImageBlock, PointBlock),
            get_items=get_image_files,
            get_y=get_ctr,
            splitter=FuncSplitter(lambda o: o.parent.name=='13'),
            batch_tfms=[*aug_transforms(size=(240,320)), 
                        Normalize.from_stats(*imagenet_stats)])
    dls = biwi.dataloaders(path)
    learn = vision_learner(dls, resnet18, y_range=(-1,1))
    with learn.distrib_ctx(in_notebook=True, sync_bn=False):
        learn.fine_tune(1)
        
notebook_launcher(train, num_processes=2)
# -

# ## fin -


