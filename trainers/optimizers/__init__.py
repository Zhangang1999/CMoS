#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-12-29

from managers.ops_manager import OpsManager
OPTIMIZERS = OpsManager('optimizer')

from .adam import Adam


__all__ = [
    'Adam',
    ]