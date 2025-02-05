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

# + active=""
# ---
# skip_exec: true
# ---

# +
#|default_exp vision.utils
# -

#|export
from __future__ import annotations
import uuid
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.vision.core import *
from fastdownload import download_url
from pathlib import Path

#|hide
from nbdev.showdoc import *


# # Vision utils
#
# > Some utils function to quickly download a bunch of images, check them and pre-resize them

#|export
def _get_downloaded_image_filename(dest, name, suffix):
    start_index = 1
    candidate_name = name

    while (dest/f"{candidate_name}{suffix}").is_file():
        candidate_name = f"{candidate_name}{start_index}"
        start_index += 1

    return candidate_name


#|export
def _download_image_inner(dest, inp, timeout=4, preserve_filename=False):
    i,url = inp
    url = url.split("?")[0]
    url_path = Path(url)
    suffix = url_path.suffix if url_path.suffix else '.jpg'
    name = _get_downloaded_image_filename(dest, url_path.stem, suffix) if preserve_filename else str(uuid.uuid4())
    try: download_url(url, dest/f"{name}{suffix}", show_progress=False, timeout=timeout)
    except Exception as e: f"Couldn't download {url}."


# +
#|hide
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    url = "https://www.fast.ai/images/jh-head.jpg"
    _download_image_inner(d, (125,url))
    test_eq(len(d.ls()), 1)

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    url = "https://www.fast.ai/images/jh-head.jpg"

    _download_image_inner(d, (125,url), preserve_filename=True)
    assert (d/'jh-head.jpg').is_file()
    assert not (d/'jh-head.jpg1').exists()

    _download_image_inner(d, (125,url), preserve_filename=True)
    assert (d/'jh-head.jpg').is_file()
    assert (d/'jh-head1.jpg').is_file()


# -

#|export
def download_images(dest, url_file=None, urls=None, max_pics=1000, n_workers=8, timeout=4, preserve_filename=False):
    "Download images listed in text file `url_file` to path `dest`, at most `max_pics`"
    if urls is None: urls = url_file.read_text().strip().split("\n")[:max_pics]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout, preserve_filename=preserve_filename),
             list(enumerate(urls)), n_workers=n_workers, threadpool=True)


#|hide
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    url_file = d/'urls.txt'
    url_file.write_text("\n".join([f"https://www.fast.ai/images/{n}" for n in "jh-head.jpg headshot-small.jpg".split()]))
    
    download_images(d, url_file, preserve_filename=True)
    assert (d/'jh-head.jpg').is_file()
    assert (d/'headshot-small.jpg').is_file()
    assert not (d/'jh-head1.jpg').exists()


#|export
def resize_to(img, targ_sz, use_min=False):
    "Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e w*h)"
    w,h = img.size
    min_sz = (min if use_min else max)(w,h)
    ratio = targ_sz/min_sz
    return int(w*ratio),int(h*ratio)


# +
class _FakeImg():
    def __init__(self, size): self.size=size

img = _FakeImg((200,500))
test_eq(resize_to(img, 400), [160,400])
test_eq(resize_to(img, 400, use_min=True), [400,1000])


# -

#|export
def verify_image(fn):
    "Confirm that `fn` can be opened"
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except: return False


#|export
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in enumerate(parallel(verify_image, fns)) if not o)


#|export
def resize_image(file, dest, src='.', max_size=None, n_channels=3, ext=None,
                 img_format=None, resample=BILINEAR, resume=False, **kwargs ):
    "Resize file to dest to max_size"
    dest = Path(dest)
    
    dest_fname = dest/file
    dest_fname.parent.mkdir(exist_ok=True, parents=True)
    file = Path(src)/file
    if resume and dest_fname.exists(): return
    if not verify_image(file): return

    img = Image.open(file)
    imgarr = np.array(img)
    img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    if ext is not None: dest_fname=dest_fname.with_suffix(ext)
    if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
        if max_size is not None:
            new_sz = resize_to(img, max_size)
            img = img.resize(new_sz, resample=resample)
        if n_channels == 3: img = img.convert("RGB")
        img.save(dest_fname, img_format, **kwargs)
    elif file != dest_fname : shutil.copy2(file, dest_fname)


file = 'puppy.jpg'
dest = Path('.')
resize_image(file, dest, src='images', max_size=400)
im = Image.open(dest/file)
test_eq(im.shape[1],400)
(dest/file).unlink()

file = 'puppy.jpg'
dest = Path('images')
resize_image(file, dest, src=dest, max_size=None)


#|export
def resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,
                  dest=Path('.'), n_channels=3, ext=None, img_format=None, resample=BILINEAR,
                  resume=None, **kwargs):
    "Resize files on path recursively to dest to max_size"
    path = Path(path)
    if resume is None and dest != Path('.'): resume=False
    os.makedirs(dest, exist_ok=True)
    files = get_image_files(path, recurse=recurse)
    files = [o.relative_to(path) for o in files]
    parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
                   img_format=img_format, resample=resample, resume=resume, **kwargs)


with tempfile.TemporaryDirectory() as d:
    dest = Path(d)/'resized_images'
    resize_images('images', max_size=100, dest=dest, max_workers=0, recurse=True)

# # Export -

#|hide
from nbdev import nbdev_export
nbdev_export()


