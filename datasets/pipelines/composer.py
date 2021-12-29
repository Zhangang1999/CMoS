import random
from typing import List
from collections.abc import Sequence
from datasets.transforms import TRANSFORMS
from utils.instantiate import instantiate_from_args

from datasets.pipelines import PIPELINES

class BaseComposer(object):
    """Base class for compose pipeline."""
    def __init__(self, transforms:List=[]):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transforms, dict):
                self.transforms.append(instantiate_from_args(transform, TRANSFORMS))
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("Transform should be a dict or callable.")

    def __call__(self, data_sample):
        raise NotImplementedError
       
    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        repr_str = "Transformations:"
        for t in self.transforms:
            repr_str += "   {}\n".format(t.__repr__())
        return repr_str

@PIPELINES.register()
class SequenceComposer(BaseComposer):
    """Composes several augmentations together as a sequence.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Usage:
        augmentations.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor(),
        ])
    """
    def __init__(self, transforms:List=[]) -> None:
        super().__init__(transforms=transforms)

    def __call__(self, data_sample):
        for t in self.transforms:
            data_sample = t(data_sample)    
        return data_sample

@PIPELINES.register()
class ChoiceComposer(BaseComposer):

    def __init__(self, transforms:List=[], num_choice:int=1):
        super().__init__(transforms=transforms)
        self._check_num_choice_if_valid(num_choice)
        self.num_choice = num_choice

    def _check_num_choice_if_valid(self, num_choice):
        assert isinstance(num_choice, int), "num_choice should be int."
        assert num_choice > 0 and num_choice < len(self.transforms)

    def __call__(self, data_sample):
        for t in random.choice(self.transforms, self.num_choice):
            data_sample = t(data_sample)
        return data_sample