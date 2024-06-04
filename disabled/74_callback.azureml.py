# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#hide
#skip
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab

# +
#all_slow
# -

#export
from fastai.basics import *
from fastai.learner import Callback

#hide
from nbdev.showdoc import *

# +
#default_exp callback.azureml
# -

# # AzureML Callback
#
# Track fastai experiments with the azure machine learning plattform.
#
# ## Prerequisites
#
# Install the azureml SDK:
#
# ```python
# pip install azureml-core
# ```
#
# ## How to use it?
#
# Import and use `AzureMLCallback` during model fitting.
#
# If you are submitting your training run with azureml SDK [ScriptRunConfig](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets), the callback will automatically detect the run and log metrics. For example:
#
# ```python
# from fastai.callback.azureml import AzureMLCallback
# learn.fit_one_cycle(epoch, lr, cbs=AzureMLCallback())
# ```
#
# If you are running an experiment manually and just want to have interactive logging of the run, use azureml's `Experiment.start_logging` to create the interactive `run`, and pass that into `AzureMLCallback`. For example:
#
# ```python
# from azureml.core import Experiment
# experiment = Experiment(workspace=ws, name='experiment_name')
# run = experiment.start_logging(outputs=None, snapshot_directory=None)
#
# from fastai.callback.azureml import AzureMLCallback
# learn.fit_one_cycle(epoch, lr, cbs=AzureMLCallback(run))
# ```
#
# If you are running an experiment on your local machine (i.e. not using `ScriptRunConfig` and not passing an `run` into the callback), it will recognize that there is no AzureML run to log to, and produce no logging output.
#
# If you are using [AzureML pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines), the `AzureMLCallback` will by default also send the same logging output to the parent run, so that metrics can easily be plotted. If you do not want this (e.g. because you have multiple training steps in a pipeline), you can disable it by setting `log_to_parent`to `False`.
#
# To save the model weights, use the usual fastai methods and save the model to the `outputs` folder, which is a "special" (for Azure) folder that is automatically tracked in AzureML.
#
# As it stands, note that if you pass the callback into your `Learner` directly, e.g.:
# ```python
# learn = Learner(dls, model, cbs=AzureMLCallback())
# ```
# …some `Learner` methods (e.g. `learn.show_results()`) might add unwanted logging into your azureml experiment runs. Adding further checks into the callback should help eliminate this – another PR needed.

#export
from azureml.core.run import Run
from azureml.exceptions import RunEnvironmentException
import warnings


# export
class AzureMLCallback(Callback):
    """
    Log losses, metrics, model architecture summary to AzureML.

    If `log_offline` is False, will only log if actually running on AzureML.
    A custom AzureML `Run` class can be passed as `azurerun`.
    If `log_to_parent` is True, will also log to the parent run, if exists (e.g. in AzureML pipelines).
    """
    order = Recorder.order+1

    def __init__(self, azurerun=None, log_to_parent=True):
        if azurerun:
            self.azurerun = azurerun
        else:
            try:
                self.azurerun = Run.get_context(allow_offline=False)
            except RunEnvironmentException:
                # running locally
                self.azurerun = None
                warnings.warn("Not running on AzureML and no azurerun passed, AzureMLCallback will be disabled.")
        self.log_to_parent = log_to_parent

    def before_fit(self):
        self._log("n_epoch", self.learn.n_epoch)
        self._log("model_class", str(type(self.learn.model)))

        try:
            summary_file = Path("outputs") / 'model_summary.txt'
            with summary_file.open("w") as f:
                f.write(repr(self.learn.model))
        except:
            print('Did not log model summary. Check if your model is PyTorch model.')

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            self._log('batch__loss', self.learn.loss.item())
            self._log('batch__train_iter', self.learn.train_iter)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    self._log(f'batch__opt.hypers.{k}', v)

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ['epoch', 'time']:
                self._log(f'epoch__{n}', v)
            if n == 'time':
                # split elapsed time string, then convert into 'seconds' to log
                m, s = str(v).split(':')
                elapsed = int(m)*60 + int(s)
                self._log(f'epoch__{n}', elapsed)

    def _log(self, metric, value):
        if self.azurerun is not None:
            self.azurerun.log(metric, value)
            if self.log_to_parent and self.azurerun.parent is not None:
                self.azurerun.parent.log(metric, value)


