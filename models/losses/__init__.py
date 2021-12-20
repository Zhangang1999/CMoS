#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-11-29

from managers.ops_manager import OpsManager
LOSSES = OpsManager('loss')

from .epe import EPELoss

__all__ = [
    'EPELoss'
]