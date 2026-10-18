"""Load this experiment's trusted checkpoints across NumPy 1.x/2.x hosts."""
import pickle
import types

import torch


class NumpyCompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_checkpoint(path):
    compatibility = types.ModuleType('spatial_numpy_compatibility_pickle')
    for key in ('load', 'loads', 'dump', 'dumps', 'Pickler'):
        setattr(compatibility, key, getattr(pickle, key))
    compatibility.Unpickler = NumpyCompatibilityUnpickler
    return torch.load(path, map_location='cpu', weights_only=False,
                      pickle_module=compatibility)
