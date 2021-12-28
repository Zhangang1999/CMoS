#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : CchenzJ (chenzejian19@email.szu.edu.cn)
# @Desc   : 
# @Date   : 2021-11-29

from managers.ops_manager import OpsManager
TRANSFORMS = OpsManager('transform')

from .base_transform import BaseTransform
from .contrast import Contrast
from .horizontal_flip import HorizonalFlip
from .vertical_flip import VerticalFlip
from .to_pil_image import ToPILImage
from .to_tensor import ToTensor
from normalization import Normalization

__all__ = [
    'BaseTransform',
    'Contrast',
    'HorizonalFlip',
    'VerticalFlip',
    'ToTensor',
    'ToPILImage',
    'Normalization'
]