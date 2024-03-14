# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Union, List

import torch

from peft.tuners.rosa.layer import RosaLayer
from peft.tuners.tuners_utils import BaseTunerLayer


from aqlm import QuantizedLinear


class AqlmLoraLinear(torch.nn.Module, RosaLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        d: float = 0.0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        spa_store_transpose: bool = True,
        rosa_dtype: str = 'bf16',
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ):
        super().__init__()
        RosaLayer.__init__(self, base_layer, "auto")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, d, lora_alpha, lora_dropout, spa_store_transpose, rosa_dtype, init_lora_weights, use_rslora)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            assert len(self.active_adapters) == 1, 'rosa only supports precisely one adapter'
            active_adapter = self.active_adapters[0]
            assert active_adapter in self.rosa_A.keys()

            if self.r[active_adapter] == 0 and not self._spa_exists(active_adapter):
                # we are collecting gradients while lora deos not exist
                # adding a dummy to the input to enable gradient propagation
                x = self._add_dummy(x)

            
            result = self.base_layer(x, *args, **kwargs)
            
            if self.r[active_adapter] > 0:
                rosa_A = self.rosa_A[active_adapter]
                rosa_B = self.rosa_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(rosa_A.weight.dtype)
                result += rosa_B(rosa_A(dropout(x))) * scaling

            if self._spa_exists(active_adapter):
                spa_module = self.rosa_spa[active_adapter]
                # x = x.to(spa_module.values.dtype)
                result += spa_module(x)
            
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep + "_aqlm"
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError("Can't merge RoSA with AQLM quantized layer.")


def dispatch_aqlm(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, QuantizedLinear):
        new_module = AqlmLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.codes

    return new_module
