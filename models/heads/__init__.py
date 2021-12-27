#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-12-20

from managers.ops_manager import OpsManager
HEADS = OpsManager('head')

from .mmsa import MMSAHead

__all__ = [
    'MMSAHead',
]