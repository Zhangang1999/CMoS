#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-12-02

from managers.ops_manager import OpsManager
HOOKS = OpsManager('hook')

from .base_hook import BaseHook
from .log_hook import LogHook
from .lr_hook import LrHook
from .metric_hook import MetricHook
from .optimizer_hook import OptimizerHook
from .visual_hook import VisualHook
from .model_hook import ModelHook

__all__ = [
    'BaseHook',
    'LogHook',
    'LrHook',
    'MetricHook',
    'OptimizerHook',
    'VisualHook',
    'ModelHook'
]