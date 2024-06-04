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

#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

# + active=""
# ---
# skip_exec: true
# ---

# +
#|default_exp callback.comet

# +
#|export
from __future__ import annotations

import tempfile

from fastai.basics import *
from fastai.learner import Callback
# -

#|hide
from nbdev.showdoc import *

# # Comet.ml
#
# > Integration with [Comet.ml](https://www.comet.ml/).

# ## Registration

# 1. Create account: [comet.ml/signup](https://www.comet.ml/signup).
# 2. Export API key to the environment variable (more help [here](https://www.comet.ml/docs/v2/guides/getting-started/quickstart/#get-an-api-key)). In your terminal run:
#
# ```
# export COMET_API_KEY='YOUR_LONG_API_TOKEN'
# ```
#
# or include it in your `./comet.config` file (**recommended**). More help is [here](https://www.comet.ml/docs/v2/guides/tracking-ml-training/jupyter-notebooks/#set-your-api-key-and-project-name).

# ## Installation

# 1. You need to install neptune-client. In your terminal run:
#
# ```
# pip install comet_ml
# ```
#
# or (alternative installation using conda). In your terminal run:
#
# ```
# conda install -c anaconda -c conda-forge -c comet_ml comet_ml
# ```

# ## How to use?

# Key is to create the callback `CometMLCallback` before you create `Learner()` like this:
#
# ```
# from fastai.callback.comet import CometMLCallback
#
# comet_ml_callback = CometCallback('PROJECT_NAME')  # specify project
#
# learn = Learner(dls, model,
#                 cbs=comet_ml_callback
#                 )
#
# learn.fit_one_cycle(1)
# ```

#|export
import comet_ml


#|export
class CometCallback(Callback):
    "Log losses, metrics, model weights, model architecture summary to neptune"
    order = Recorder.order + 1

    def __init__(self, project_name, log_model_weights=True):
        self.log_model_weights = log_model_weights
        self.keep_experiment_running = keep_experiment_running
        self.project_name = project_name
        self.experiment = None

    def before_fit(self):
        try:
            self.experiment = comet_ml.Experiment(project_name=self.project_name)
        except ValueError:
            print("No active experiment")

        try:
            self.experiment.log_parameter("n_epoch", str(self.learn.n_epoch))
            self.experiment.log_parameter("model_class", str(type(self.learn.model)))
        except:
            print(f"Did not log all properties.")

        try:
            with tempfile.NamedTemporaryFile(mode="w") as f:
                with open(f.name, "w") as g:
                    g.write(repr(self.learn.model))
                self.experiment.log_asset(f.name, "model_summary.txt")
        except:
            print("Did not log model summary. Check if your model is PyTorch model.")

        if self.log_model_weights and not hasattr(self.learn, "save_model"):
            print(
                "Unable to log model to Comet.\n",
            )

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            self.experiment.log_metric("batch__smooth_loss", self.learn.smooth_loss)
            self.experiment.log_metric("batch__loss", self.learn.loss)
            self.experiment.log_metric("batch__train_iter", self.learn.train_iter)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    self.experiment.log_metric(f"batch__opt.hypers.{k}", v)

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ["epoch", "time"]:
                self.experiment.log_metric(f"epoch__{n}", v)
            if n == "time":
                self.experiment.log_text(f"epoch__{n}", str(v))

        # log model weights
        if self.log_model_weights and hasattr(self.learn, "save_model"):
            if self.learn.save_model.every_epoch:
                _file = join_path_file(
                    f"{self.learn.save_model.fname}_{self.learn.save_model.epoch}",
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            else:
                _file = join_path_file(
                    self.learn.save_model.fname,
                    self.learn.path / self.learn.model_dir,
                    ext=".pth",
                )
            self.experiment.log_asset(_file)

    def after_fit(self):
        try:
            self.experiment.end()
        except:
            print("No neptune experiment to stop.")


show_doc(CometCallback)


