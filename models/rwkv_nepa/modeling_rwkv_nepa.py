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
"""PyTorch RwkvNepa model."""

import collections.abc
from dataclasses import dataclass
import warnings
from typing import Optional, Union

import torch
from torch import nn

from transformers.modeling_outputs import ImageClassifierOutput, ModelOutput
from transformers.models.rwkv.modeling_rwkv import RwkvModel, RwkvPreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.generic import can_return_tuple

from .configuration_rwkv_nepa import RwkvNepaConfig
from ..vit_nepa.modeling_vit_nepa import (
    ViTNepaEmbeddings,
    augment_patches_center_coordinates,
    get_patches_center_coordinates,
    prediction_loss,
)


logger = logging.get_logger(__name__)


class RwkvNepaCoordPositionEmbedding(nn.Module):
    def __init__(self, config: RwkvNepaConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(2, config.hidden_size, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches_h = height // patch_size[0]
        num_patches_w = width // patch_size[1]

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            coords = get_patches_center_coordinates(num_patches_h, num_patches_w, dtype=torch.float32, device=device)
            if self.training:
                coords = augment_patches_center_coordinates(
                    coords,
                    shift=self.config.pos_embed_shift,
                    jitter=self.config.pos_embed_jitter,
                    rescale=self.config.pos_embed_rescale,
                )
            pos = self.proj(coords)  # (num_patches, hidden)

        return pos.to(dtype=pixel_values.dtype)


@dataclass
@auto_docstring(
    custom_intro="""
    Base model output for RWKV-NEPA that includes both the predicted hidden states and the clean input embeddings.
    """
)
class RwkvNepaOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    input_embedding: Optional[torch.FloatTensor] = None
    state: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Output type for NEPA pretraining that exposes a single embedding prediction loss.
    """
)
class EmbeddedModelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@auto_docstring
class RwkvNepaPreTrainedModel(RwkvPreTrainedModel):
    config: RwkvNepaConfig
    base_model_prefix = "rwkv_nepa"
    main_input_name = "pixel_values"

    _no_split_modules = ["ViTNepaEmbeddings", "RwkvNepaCoordPositionEmbedding", "RwkvModel"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            std = float(getattr(self.config, "initializer_range", 0.02))
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=std).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
            return
        if isinstance(module, ViTNepaEmbeddings):
            std = float(getattr(self.config, "initializer_range", 0.02))
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data.to(torch.float32), mean=0.0, std=std).to(module.cls_token.dtype)
            if module.mask_token is not None:
                module.mask_token.data.zero_()
            return

        super()._init_weights(module)

    def _set_trainable(self, trainable: str = "timemix") -> None:
        """
        Freeze all parameters, then selectively unfreeze a subset.

        Supported values:
          - "timemix": unfreeze RWKV TimeMix blocks (attention submodules) + all LayerNorm parameters
          - "all": unfreeze everything
          - "none": keep everything frozen
        """
        trainable = (trainable or "").lower().strip()

        self.requires_grad_(False)

        if trainable in ("", "timemix"):
            core = getattr(self, "rwkv_nepa", None)
            if core is None:
                core = self
            rwkv = getattr(core, "rwkv", None)
            if rwkv is None or not hasattr(rwkv, "blocks"):
                raise ValueError("Could not locate RWKV backbone to unfreeze timemix parameters.")
            for block in rwkv.blocks:
                block.attention.requires_grad_(True)
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
            return

        if trainable == "all":
            self.requires_grad_(True)
            return

        if trainable in ("none", "frozen"):
            return

        raise ValueError(f"Unsupported trainable={trainable!r}. Use timemix|all|none.")


@auto_docstring
class RwkvNepaModel(RwkvNepaPreTrainedModel):
    def __init__(self, config: RwkvNepaConfig, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        # Reuse ViTNepa embedding pipeline (patch conv + CLS + optional mask token + dropout).
        self.embeddings = ViTNepaEmbeddings(config, use_mask_token=use_mask_token)
        self.pos_embedding = None
        if getattr(config, "pos_embed_type", "none") == "coord_linear":
            self.pos_embedding = RwkvNepaCoordPositionEmbedding(config)
        elif getattr(config, "pos_embed_type", "none") not in ("none", None):
            raise ValueError(f"Unsupported pos_embed_type: {config.pos_embed_type!r}")

        self.rwkv = RwkvModel(config)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embeddings

    @property
    def layernorm(self) -> nn.Module:
        # Backward-compat (ViT-like): expose final LayerNorm without duplicating weights in the state_dict.
        return self.rwkv.ln_out

    @property
    def encoder(self):
        # Backward-compat (ViT-like): expose RWKV blocks at `.encoder.layer` without registering duplicate modules.
        import types

        return types.SimpleNamespace(layer=self.rwkv.blocks)

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        state: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        is_pretraining: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RwkvNepaOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if head_mask is not None:
            warnings.warn("`head_mask` is unused for RwkvNepaModel (RWKV has no attention heads).")
        if interpolate_pos_encoding:
            warnings.warn("`interpolate_pos_encoding` is unused for RwkvNepaModel (patch embedding requires fixed size).")

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_input, embedding_clean = self.embeddings(
            pixel_values,
            position_ids=position_ids,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=False,
        )
        if self.pos_embedding is not None:
            pos = self.pos_embedding(pixel_values)  # (num_patches, hidden)
            embedding_input[:, -pos.shape[0] :, :] = embedding_input[:, -pos.shape[0] :, :] + pos.unsqueeze(0)

        rwkv_outputs = self.rwkv(
            inputs_embeds=embedding_input,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return RwkvNepaOutput(
            last_hidden_state=rwkv_outputs.last_hidden_state,
            input_embedding=embedding_clean,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )


class RwkvNepaForPreTraining(RwkvNepaPreTrainedModel):
    def __init__(self, config: RwkvNepaConfig):
        super().__init__(config)
        self.rwkv_nepa = RwkvNepaModel(config)
        self.post_init()

    @property
    def vit_nepa(self) -> RwkvNepaModel:
        # Compatibility alias (do not register as a submodule to avoid duplicating weights in the state_dict).
        return self.rwkv_nepa

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddedModelingOutput:
        outputs: RwkvNepaOutput = self.rwkv_nepa(
            pixel_values=pixel_values,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            is_pretraining=True,
            return_dict=True,
            **kwargs,
        )

        embedded_loss = prediction_loss(outputs.input_embedding, outputs.last_hidden_state)

        return EmbeddedModelingOutput(
            loss=embedded_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class RwkvNepaForImageClassification(RwkvNepaPreTrainedModel):
    def __init__(self, config: RwkvNepaConfig):
        super().__init__(config)
        self.add_pooling_layer = config.add_pooling_layer

        image_size = config.image_size if isinstance(config.image_size, collections.abc.Iterable) else (config.image_size, config.image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        self.num_image_tokens = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.num_labels = config.num_labels
        self.rwkv_nepa = RwkvNepaModel(config)
        self.fc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon) if config.add_pooling_layer else None

        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @property
    def vit_nepa(self) -> RwkvNepaModel:
        # Compatibility alias (do not register as a submodule to avoid duplicating weights in the state_dict).
        return self.rwkv_nepa

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        outputs: RwkvNepaOutput = self.rwkv_nepa(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        if self.add_pooling_layer:
            image_tokens = sequence_output[:, -self.num_image_tokens :, :]
            pooled_output = image_tokens.mean(dim=1)
            pooled_output = self.fc_norm(pooled_output)
        else:
            pooled_output = sequence_output[:, -1, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "RwkvNepaConfig",
    "RwkvNepaModel",
    "RwkvNepaPreTrainedModel",
    "RwkvNepaForPreTraining",
    "RwkvNepaForImageClassification",
]
