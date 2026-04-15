# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file exceam in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RwkvNepa model configuration"""

import collections.abc
from typing import Optional, Union

from transformers.models.rwkv.configuration_rwkv import RwkvConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class RwkvNepaConfig(RwkvConfig):
    r"""
    This is the configuration class to store the configuration of a [`RwkvNepaModel`]. It is used to instantiate a
    RWKV-based NEPA vision model according to the specified arguments, defining the model architecture.

    This config inherits from [`RwkvConfig`] and adds vision-specific fields (image/patch embedding + 2D coordinates).

    Args:
        vocab_size (`int`, *optional*, defaults to 1):
            RWKV vocabulary size. For NEPA-vision we typically use `inputs_embeds`, so this can be small.
        context_length (`int`, *optional*, defaults to `None`):
            RWKV context length. If `None`, it is derived from `image_size` and `patch_size` as
            `(H//P_h)*(W//P_w) + 1` (+1 for the CLS token).
        hidden_size (`int`, *optional*, defaults to 768):
            Hidden size for both patch embeddings and RWKV.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of RWKV blocks.
        attention_hidden_size (`int`, *optional*, defaults to `None`):
            RWKV attention hidden size. If `None`, defaults to `hidden_size` (same as [`RwkvConfig`]).
        intermediate_size (`int`, *optional*, defaults to `None`):
            RWKV MLP size. If `None`, defaults to `4 * hidden_size` (same as [`RwkvConfig`]).
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon for layer norm layers used by RWKV.
        rescale_every (`int`, *optional*, defaults to 6):
            RWKV rescaling interval.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether to default to returning RWKV recurrent `state`. For training, this is typically `False`.
        image_size (`int` or `tuple[int, int]`, *optional*, defaults to 224):
            Input image resolution (H, W) or a single int for square images.
        patch_size (`int` or `tuple[int, int]`, *optional*, defaults to 14):
            Patch size (P_h, P_w) or a single int for square patches.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout probability applied after building the input embedding sequence.
        add_pooling_layer (`bool`, *optional*, defaults to `False`):
            Whether to use mean pooling over image tokens for classification.
        pos_embed_type (`str`, *optional*, defaults to `"none"`):
            How to inject 2D position information for patch tokens. Supported: `"none"`, `"coord_linear"`.
        pos_embed_shift (`float`, *optional*, defaults to `None`):
            Random shift magnitude for patch coordinates in training augmentation.
        pos_embed_jitter (`float`, *optional*, defaults to `None`):
            Random jitter magnitude for patch coordinates in training augmentation.
        pos_embed_rescale (`float`, *optional*, defaults to 2.0):
            Random rescale magnitude for patch coordinates in training augmentation.
        kwargs:
            Additional keyword arguments passed to [`RwkvConfig`]/[`PretrainedConfig`].
    """

    model_type = "rwkv_nepa"

    def __init__(
        self,
        vocab_size: int = 1,
        context_length: Optional[int] = None,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        attention_hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        layer_norm_epsilon: float = 1e-5,
        rescale_every: int = 6,
        use_cache: bool = False,
        image_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 14,
        num_channels: int = 3,
        hidden_dropout_prob: float = 0.0,
        add_pooling_layer: bool = False,
        pos_embed_type: str = "none",
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = 2.0,
        **kwargs,
    ):
        if num_hidden_layers < 2:
            raise ValueError(
                "`num_hidden_layers` must be >= 2 for Transformers `RwkvModel` (its weight init assumes >=2 layers)."
            )

        image_size_hw = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size_hw = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        if context_length is None:
            num_patches = (image_size_hw[0] // patch_size_hw[0]) * (image_size_hw[1] // patch_size_hw[1])
            context_length = int(num_patches + 1)  # +1 for CLS token

        super().__init__(
            vocab_size=vocab_size,
            context_length=context_length,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            attention_hidden_size=attention_hidden_size,
            intermediate_size=intermediate_size,
            layer_norm_epsilon=layer_norm_epsilon,
            rescale_every=rescale_every,
            use_cache=use_cache,
            **kwargs,
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.add_pooling_layer = add_pooling_layer

        self.pos_embed_type = pos_embed_type
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale


__all__ = ["RwkvNepaConfig"]
