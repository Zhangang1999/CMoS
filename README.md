# Introduction

Here, is an introduction for the CMoS repo., including:

1. MANAGERS
2. TRAINERS
3. MODELS
4. METRICS
5. INTERFACE
6. MSIC

## MANAGERS

[FileManager](managers/file_manager.py), to manager the paths/files corresponding to model name for exps.

[OpsManager](managers/ops_manager.py), to manager the operations for building the model.

## TRAINERS

[BaseTrainer](trainers/base_trainer.py), the base trainer as template class with some basic functions.

[EpochTrainer](trainers/epoch_trainer.py), the trainer in epoch training fashion.

### HOOKS

[BaseHook](trainers/hooks/base_hook.py), the base hook as template class with some basic functions.

[LogHook](trainers/hooks/log_hook.py), the hook calling to log some infomations.

[LrHook](trainers/hooks/lr_hook.py), the hook calling to change the learning rate.

[MetricHook](trainers/hooks/metric_hook.py), the hook calling to calculate the metrics.

[OptimizerHook](trainers/hooks/optimizer_hook.py), the hook calling to update the optimizer.

[VisualHook](trainers/hooks/visual_hook.py), the hook calling to add info to visualizer.

## MODELS

[BaseModel](models/base_model.py), the base model as template class with some basic functions.

### BACKBONES

[FlowNetS, FlowNetC](models/backbones/flownet.py), the flownet backbone for optical flow estimiation.

### HEADS

[MMSAHead](models/heads/mmsa.py), the head for Cardiac motion scoring with multi-scale motion spatial attention.

### LOSSES

[MMSALoss](models/losses/mmsa.py), the losses for the MMSA model.

[EPELoss](models/losses/epe.py), the losses for the expect prediction error loss, using in flownet.

### OPS

### POSTPROCESS

## METRICS

## INTERFACES

### TRAIN

### TEST

### INFER

## MSIC
