#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-11-29

from managers.ops_manager import OpsManager
PIPELINES = OpsManager('pipeline')

from .composer import BaseComposer, SequenceComposer, ChoiceComposer
from .fetcher import Fetcher
from .loader import Loader