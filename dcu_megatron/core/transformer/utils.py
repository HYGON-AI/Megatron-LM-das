from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SubmoduleCallables:
    """
    Holds references to forward, dgrad, and dw (weight-grad) callables
    for a particular submodule.
    """

    forward: Optional[Callable] = None
    backward: Optional[Callable] = None
    dgrad: Optional[Callable] = None
    dw: Optional[Callable] = None


@dataclass
class TransformerLayerSubmoduleCallables:
    """
    Collects the SubmoduleMethods for each of the submodules:
    'attention', 'dispatch', 'mlp', 'combine'.
    """

    attention: SubmoduleCallables
    dispatch: SubmoduleCallables
    mlp: SubmoduleCallables
    combine: SubmoduleCallables
    post_combine: SubmoduleCallables

    def as_array(self):
        return [self.attention, self.dispatch, self.mlp, self.combine, self.post_combine]
