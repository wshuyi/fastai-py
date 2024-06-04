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
#|default_exp data.external
# -

#|export
from __future__ import annotations
from fastai.torch_basics import *
from fastdownload import FastDownload
from functools import lru_cache
import fastai.data


# # External data
# > Helper functions to download the fastai datasets

# To download any of the datasets or pretrained weights, simply run `untar_data` by passing any dataset name mentioned above like so: 
#
# ```python 
# path = untar_data(URLs.PETS)
# path.ls()
#
# >> (#7393) [Path('/home/ubuntu/.fastai/data/oxford-iiit-pet/images/keeshond_34.jpg'),...]
# ```
#
# To download model pretrained weights: 
# ```python 
# path = untar_data(URLs.WT103_BWD)
# path.ls()
#
# >> (#2) [Path('/home/ubuntu/.fastai/data/wt103-bwd/itos_wt103.pkl'),Path('/home/ubuntu/.fastai/data/wt103-bwd/lstm_bwd.pth')]
# ```

# ## Datasets

#  A complete list of datasets that are available by default inside the library are: 

# ### Main datasets

# 1.    **ADULT_SAMPLE**: A small of the [adults dataset](https://archive.ics.uci.edu/ml/datasets/Adult) to  predict whether income exceeds $50K/yr based on census data. 
# -    **BIWI_SAMPLE**: A [BIWI kinect headpose database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). The dataset contains over 15K images of 20 people (6 females and 14 males - 4 people were recorded twice). For each frame, a depth image, the corresponding rgb image (both 640x480 pixels), and the annotation is provided. The head pose range covers about +-75 degrees yaw and +-60 degrees pitch. 
# 1.    **CIFAR**: The famous [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.      
# 1.    **COCO_SAMPLE**: A sample of the [coco dataset](http://cocodataset.org/#home) for object detection. 
# 1.    **COCO_TINY**: A tiny version of the [coco dataset](http://cocodataset.org/#home) for object detection.
# -    **HUMAN_NUMBERS**: A synthetic dataset consisting of human number counts in text such as one, two, three, four.. Useful for experimenting with Language Models.
# -    **IMDB**: The full [IMDB sentiment analysis dataset](https://ai.stanford.edu/~amaas/data/sentiment/).          
#
# -    **IMDB_SAMPLE**: A sample of the full [IMDB sentiment analysis dataset](https://ai.stanford.edu/~amaas/data/sentiment/). 
# -    **ML_SAMPLE**: A movielens sample dataset for recommendation engines to recommend movies to users.            
# -    **ML_100k**: The movielens 100k dataset for recommendation engines to recommend movies to users.             
# -    **MNIST_SAMPLE**: A sample of the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.        
# -    **MNIST_TINY**: A tiny version of the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.                   
# -    **MNIST_VAR_SIZE_TINY**:  
# -    **PLANET_SAMPLE**: A sample of the planets dataset from the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).
# -    **PLANET_TINY**: A tiny version  of the planets dataset from the Kaggle competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) for faster experimentation and prototyping.
# -    **IMAGENETTE**: A smaller version of the [imagenet dataset](http://www.image-net.org/) pronounced just like 'Imagenet', except with a corny inauthentic French accent. 
# -    **IMAGENETTE_160**: The 160px version of the Imagenette dataset.      
# -    **IMAGENETTE_320**: The 320px version of the Imagenette dataset. 
# -    **IMAGEWOOF**: Imagewoof is a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds.
# -    **IMAGEWOOF_160**: 160px version of the ImageWoof dataset.        
# -    **IMAGEWOOF_320**: 320px version of the ImageWoof dataset.
# -    **IMAGEWANG**: Imagewang contains Imagenette and Imagewoof combined, but with some twists that make it into a tricky semi-supervised unbalanced classification problem
# -    **IMAGEWANG_160**: 160px version of Imagewang.        
# -    **IMAGEWANG_320**: 320px version of Imagewang. 

# ### Kaggle competition datasets

# 1. **DOGS**: Image dataset consisting of dogs and cats images from [Dogs vs Cats kaggle competition](https://www.kaggle.com/c/dogs-vs-cats). 

# ### Image Classification datasets

# 1.    **CALTECH_101**: Pictures of objects belonging to 101 categories. About 40 to 800 images per category. Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato.
# 1.    **CARS**: The [Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) contains 16,185 images of 196 classes of cars.   
# 1.    **CIFAR_100**: The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class.   
# 1.    **CUB_200_2011**: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations
# 1.    **FLOWERS**: 17 category [flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/) by gathering images from various websites.
# 1.    **FOOD**:         
# 1.    **MNIST**: [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consisting of handwritten digits.      
# 1.    **PETS**: A 37 category [pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) with roughly 200 images for each class.

# ### NLP datasets

# 1.    **AG_NEWS**: The AG News corpus consists of news articles from the AG’s corpus of news articles on the web pertaining to the 4 largest classes. The dataset contains 30,000 training and 1,900 testing examples for each class.
# 1.    **AMAZON_REVIEWS**: This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.
# 1.    **AMAZON_REVIEWS_POLARITY**: Amazon reviews dataset for sentiment analysis.
# 1.    **DBPEDIA**: The DBpedia ontology dataset contains 560,000 training samples and 70,000 testing samples for each of 14 nonoverlapping classes from DBpedia. 
# 1.    **MT_ENG_FRA**: Machine translation dataset from English to French.
# 1.    **SOGOU_NEWS**: [The Sogou-SRR](http://www.thuir.cn/data-srr/) (Search Result Relevance) dataset was constructed to support researches on search engine relevance estimation and ranking tasks.
# 1.    **WIKITEXT**: The [WikiText language modeling dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia.  
# 1.    **WIKITEXT_TINY**: A tiny version of the WIKITEXT dataset.
# 1.    **YAHOO_ANSWERS**: YAHOO's question answers dataset.
# 1.    **YELP_REVIEWS**: The [Yelp dataset](https://www.yelp.com/dataset) is a subset of YELP businesses, reviews, and user data for use in personal, educational, and academic purposes
# 1.    **YELP_REVIEWS_POLARITY**: For sentiment classification on YELP reviews.

# ### Image localization datasets

# 1.    **BIWI_HEAD_POSE**: A [BIWI kinect headpose database](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). The dataset contains over 15K images of 20 people (6 females and 14 males - 4 people were recorded twice). For each frame, a depth image, the corresponding rgb image (both 640x480 pixels), and the annotation is provided. The head pose range covers about +-75 degrees yaw and +-60 degrees pitch. 
# 1.    **CAMVID**: Consists of driving labelled dataset for segmentation type models.
# 1.    **CAMVID_TINY**: A tiny camvid dataset for segmentation type models.
# 1.    **LSUN_BEDROOMS**: [Large-scale Image Dataset](https://arxiv.org/abs/1506.03365) using Deep Learning with Humans in the Loop
# 1.    **PASCAL_2007**: [Pascal 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) to recognize objects from a number of visual object classes in realistic scenes.
# 1.    **PASCAL_2012**: [Pascal 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) to recognize objects from a number of visual object classes in realistic scenes.

# ### Audio classification

# 1. **MACAQUES**: [7285 macaque coo calls](https://datadryad.org/stash/dataset/doi:10.5061/dryad.7f4p9) across 8 individuals from [Distributed acoustic cues for caller identity in macaque vocalization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4806230).
# 2. **ZEBRA_FINCH**: [3405 zebra finch calls](https://ndownloader.figshare.com/articles/11905533/versions/1) classified [across 11 call types](https://link.springer.com/article/10.1007/s10071-015-0933-6). Additional labels include name of individual making the vocalization and its age.

# ### Medical imaging datasets

# 1. **SIIM_SMALL**: A smaller version of the [SIIM dataset](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview) where the objective is to classify pneumothorax from a set of chest radiographic images.
# 2. **TCGA_SMALL**: A smaller version of the [TCGA-OV dataset](http://doi.org/10.7937/K9/TCIA.2016.NDO1MDFQ) with subcutaneous and visceral fat segmentations. Citations:
#
#     Holback, C., Jarosz, R., Prior, F., Mutch, D. G., Bhosale, P., Garcia, K., … Erickson, B. J. (2016). Radiology Data from The Cancer Genome Atlas Ovarian Cancer [TCGA-OV] collection. The Cancer Imaging Archive. [paper](http://doi.org/10.7937/K9/TCIA.2016.NDO1MDFQ)
#
#     Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. [paper](https://link.springer.com/article/10.1007/s10278-013-9622-7)

# ### Pretrained models

# 1.    **OPENAI_TRANSFORMER**: The GPT2 Transformer pretrained weights.
# 1.    **WT103_FWD**: The WikiText-103 forward language model weights.
# 1.    **WT103_BWD**: The WikiText-103 backward language model weights.

# ## Config

#|export
@lru_cache(maxsize=None)
def fastai_cfg() -> Config: # Config that contains default download paths for `data`, `model`, `storage` and `archive`
    "`Config` object for fastai's `config.ini`"
    return Config(Path(os.getenv('FASTAI_HOME', '~/.fastai')), 'config.ini', create=dict(
        data = 'data', archive = 'archive', storage = 'tmp', model = 'models'))


# This is a basic `Config` file that consists of `data`, `model`, `storage` and `archive`. 
# All future downloads occur at the paths defined in the config file based on the type of download. For example, all future fastai datasets are downloaded to the `data` while all pretrained model weights are download to `model` unless the default download location is updated. The config file directory is defined by enviromental variable `FASTAI_HOME` if it exists, otherwise it is set to `~/.fastai`.

cfg = fastai_cfg()
cfg.data,cfg.path('data')


#|export
def fastai_path(folder:str) -> Path: 
    "Local path to `folder` in `Config`"
    return fastai_cfg().path(folder)


fastai_path('archive')


# ## URLs -

#|export
class URLs():
    "Global constants for dataset and model URLs."
    LOCAL_PATH = Path.cwd()
    MDL = 'http://files.fast.ai/models/'
    GOOGLE = 'https://storage.googleapis.com/'
    S3  = 'https://s3.amazonaws.com/fast-ai-'
    URL = f'{S3}sample/'

    S3_IMAGE    = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_AUDI     = f'{S3}audio/'
    S3_NLP      = f'{S3}nlp/'
    S3_COCO     = f'{S3}coco/'
    S3_MODEL    = f'{S3}modelzoo/'

    # main datasets
    ADULT_SAMPLE        = f'{URL}adult_sample.tgz'
    BIWI_SAMPLE         = f'{URL}biwi_sample.tgz'
    CIFAR               = f'{URL}cifar10.tgz'
    COCO_SAMPLE         = f'{S3_COCO}coco_sample.tgz'
    COCO_TINY           = f'{S3_COCO}coco_tiny.tgz'
    HUMAN_NUMBERS       = f'{URL}human_numbers.tgz'
    IMDB                = f'{S3_NLP}imdb.tgz'
    IMDB_SAMPLE         = f'{URL}imdb_sample.tgz'
    ML_SAMPLE           = f'{URL}movie_lens_sample.tgz'
    ML_100k             = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    MNIST_SAMPLE        = f'{URL}mnist_sample.tgz'
    MNIST_TINY          = f'{URL}mnist_tiny.tgz'
    MNIST_VAR_SIZE_TINY = f'{S3_IMAGE}mnist_var_size_tiny.tgz'
    PLANET_SAMPLE       = f'{URL}planet_sample.tgz'
    PLANET_TINY         = f'{URL}planet_tiny.tgz'
    IMAGENETTE          = f'{S3_IMAGE}imagenette2.tgz'
    IMAGENETTE_160      = f'{S3_IMAGE}imagenette2-160.tgz'
    IMAGENETTE_320      = f'{S3_IMAGE}imagenette2-320.tgz'
    IMAGEWOOF           = f'{S3_IMAGE}imagewoof2.tgz'
    IMAGEWOOF_160       = f'{S3_IMAGE}imagewoof2-160.tgz'
    IMAGEWOOF_320       = f'{S3_IMAGE}imagewoof2-320.tgz'
    IMAGEWANG           = f'{S3_IMAGE}imagewang.tgz'
    IMAGEWANG_160       = f'{S3_IMAGE}imagewang-160.tgz'
    IMAGEWANG_320       = f'{S3_IMAGE}imagewang-320.tgz'

    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats.tgz'

    # image classification datasets
    CALTECH_101  = f'{S3_IMAGE}caltech_101.tgz'
    CARS         = f'{S3_IMAGE}stanford-cars.tgz'
    CIFAR_100    = f'{S3_IMAGE}cifar100.tgz'
    CUB_200_2011 = f'{S3_IMAGE}CUB_200_2011.tgz'
    FLOWERS      = f'{S3_IMAGE}oxford-102-flowers.tgz'
    FOOD         = f'{S3_IMAGE}food-101.tgz'
    MNIST        = f'{S3_IMAGE}mnist_png.tgz'
    PETS         = f'{S3_IMAGE}oxford-iiit-pet.tgz'

    # NLP datasets
    AG_NEWS                 = f'{S3_NLP}ag_news_csv.tgz'
    AMAZON_REVIEWS          = f'{S3_NLP}amazon_review_full_csv.tgz'
    AMAZON_REVIEWS_POLARITY = f'{S3_NLP}amazon_review_polarity_csv.tgz'
    DBPEDIA                 = f'{S3_NLP}dbpedia_csv.tgz'
    MT_ENG_FRA              = f'{S3_NLP}giga-fren.tgz'
    SOGOU_NEWS              = f'{S3_NLP}sogou_news_csv.tgz'
    WIKITEXT                = f'{S3_NLP}wikitext-103.tgz'
    WIKITEXT_TINY           = f'{S3_NLP}wikitext-2.tgz'
    YAHOO_ANSWERS           = f'{S3_NLP}yahoo_answers_csv.tgz'
    YELP_REVIEWS            = f'{S3_NLP}yelp_review_full_csv.tgz'
    YELP_REVIEWS_POLARITY   = f'{S3_NLP}yelp_review_polarity_csv.tgz'

    # Image localization datasets
    BIWI_HEAD_POSE     = f"{S3_IMAGELOC}biwi_head_pose.tgz"
    CAMVID             = f'{S3_IMAGELOC}camvid.tgz'
    CAMVID_TINY        = f'{URL}camvid_tiny.tgz'
    LSUN_BEDROOMS      = f'{S3_IMAGE}bedroom.tgz'
    PASCAL_2007        = f'{S3_IMAGELOC}pascal_2007.tgz'
    PASCAL_2012        = f'{S3_IMAGELOC}pascal_2012.tgz'

    # Audio classification datasets
    MACAQUES           = f'{GOOGLE}ml-animal-sounds-datasets/macaques.zip'
    ZEBRA_FINCH        = f'{GOOGLE}ml-animal-sounds-datasets/zebra_finch.zip'

    # Medical Imaging datasets
    #SKIN_LESION        = f'{S3_IMAGELOC}skin_lesion.tgz'
    SIIM_SMALL         = f'{S3_IMAGELOC}siim_small.tgz'
    TCGA_SMALL         = f'{S3_IMAGELOC}tcga_small.tgz'

    #Pretrained models
    OPENAI_TRANSFORMER = f'{S3_MODEL}transformer.tgz'
    WT103_FWD          = f'{S3_MODEL}wt103-fwd.tgz'
    WT103_BWD          = f'{S3_MODEL}wt103-bwd.tgz'

    def path(
        url:str='.', # File to download
        c_key:str='archive' # Key in `Config` where to save URL
    ) -> Path:
        "Local path where to download based on `c_key`"
        fname = url.split('/')[-1]
        local_path = URLs.LOCAL_PATH/('models' if c_key=='model' else 'data')/fname
        if local_path.exists(): return local_path
        return fastai_path(c_key)/fname


# The default local path is at `~/.fastai/archive/` but this can be updated by passing a different `c_key`. Note: `c_key` should be one of `'archive', 'data', 'model', 'storage'`.

url = URLs.PETS
local_path = URLs.path(url)
test_eq(local_path.parent, fastai_path('archive'))
local_path

local_path = URLs.path(url, c_key='model')
test_eq(local_path.parent, fastai_path('model'))
local_path


# ## untar_data -

#|export
def untar_data(
    url:str, # File to download
    archive:Path=None, # Optional override for `Config`'s `archive` key
    data:Path=None, # Optional override for `Config`'s `data` key
    c_key:str='data', # Key in `Config` where to extract file
    force_download:bool=False, # Setting to `True` will overwrite any existing copy of data
    base:str='~/.fastai' # Directory containing config file and base of relative paths
) -> Path: # Path to extracted file(s)
    "Download `url` using `FastDownload.get`"
    d = FastDownload(fastai_cfg(), module=fastai.data, archive=archive, data=data, base=base)
    return d.get(url, force=force_download, extract_key=c_key)


# `untar_data` is a thin wrapper for `FastDownload.get`. It downloads and extracts `url`, by default to subdirectories of `~/.fastai` (see `fastai_cfg` for details), and returns the path to the extracted data. Setting the `force_download` flag to 'True' will overwrite any existing copy of the data already present. For an explanation of the `c_key` parameter, see `URLs`.

untar_data(URLs.MNIST_SAMPLE)

# +
#|hide
#Check all URLs are in the download_checks.py file and match for downloaded archives
# from fastdownload import read_checks
# fd = FastDownload(fastai_cfg(), module=fastai.data)
# _whitelist = "MDL LOCAL_PATH URL WT103_BWD WT103_FWD GOOGLE".split()
# checks = read_checks(fd.module)

# for d in dir(URLs): 
#     if d.upper() == d and not d.startswith("S3") and not d in _whitelist: 
#         url = getattr(URLs, d)
#         assert url in checks,f"""{d} is not in the check file for all URLs.
# To fix this, you need to run the following code in this notebook before making a PR (there is a commented cell for this below):
# url = URLs.{d}
# fd.get(url, force=True)
# fd.update(url)
# """
#         f = fd.download(url)
#         assert fd.check(url, f),f"""The log we have for {d} in checks does not match the actual archive.
# To fix this, you need to run the following code in this notebook before making a PR (there is a commented cell for this below):
# url = URLs.{d}
# _add_check(url, URLs.path(url))
# """
# -

# ## Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


