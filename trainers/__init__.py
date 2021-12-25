#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-12-02


from managers.ops_manager import OpsManager
TRAINERS = OpsManager('trainer')

from .epoch_trainer import EpochTrainer

__all__ = [
    'EpochTrainer',
]