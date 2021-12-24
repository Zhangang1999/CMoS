#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : Store some metrics.
# @Date   : 2021-11-29

from managers.ops_manager import OpsManager
METRICS = OpsManager('metric')

from .base_metric import BaseMetric
from .acc import ACC