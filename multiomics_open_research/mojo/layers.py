# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


class ConvBlock(hk.Module):
    """
    Conv Block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 1,
        layer_norm_axis: int = -1,
        name: Optional[str] = None,
    ):
        """
        Args:
            dim: input dimension.
            dim_out: output dimension.
            kernel_size: kernel's size.
            name: model's name.
        """
        super().__init__(name=name)
        self._dim = dim
        self._dim_out = dim_out
        self._kernel_size = kernel_size
        self._layer_norm_axis = layer_norm_axis

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = hk.Conv1D(
            output_channels=self._dim if self._dim_out is None else self._dim_out,
            kernel_shape=self._kernel_size,
            padding="SAME",
            data_format="NWC",
        )

        layer_norm = hk.LayerNorm(
            axis=self._layer_norm_axis,
            create_scale=True,
            create_offset=True,
            eps=1e-5,
            param_axis=self._layer_norm_axis,
        )

        x = layer_norm(x)
        x = conv(x)
        x = jax.nn.gelu(x)
        return x


class ResidualConvBlock(hk.Module):
    """
    Conv Block with Residual connection.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 1,
        name: Optional[str] = None,
    ):
        """
        Args:
            dim: input dimension.
            dim_out: output dimension.
            kernel_size: kernel's size.
            name: model's name.
        """
        super().__init__(name=name)
        self._dim = dim
        self._dim_out = dim_out
        self._kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv_block = ConvBlock(
            dim=self._dim,
            dim_out=self._dim_out,
            kernel_size=self._kernel_size,
        )
        return x + conv_block(x)


class ResidualDeConvBlock(hk.Module):
    """
    Conv Block with Residual connection.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        name: Optional[str] = None,
    ):
        """
        Args:
            dim: input dimension.
            dim_out: output dimension.
            kernel_size: kernel's size.
            stride: kernel's stride.
            name: model's name.
        """
        super().__init__(name=name)
        self._dim = dim
        self._dim_out = dim_out
        self._kernel_size = kernel_size
        self._stride = stride

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv_block = DeConvBlock(
            dim=self._dim,
            dim_out=self._dim_out,
            kernel_size=self._kernel_size,
            stride=self._stride,
        )
        return x + conv_block(x)


class DeConvBlock(hk.Module):
    """
    Conv Block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        layer_norm_axis: int = -1,
        name: Optional[str] = None,
    ):
        """
        Args:
            dim: input dimension.
            dim_out: output dimension.
            kernel_size: kernel's size.
            stride: kernel's stride.
            name: model's name.
        """
        super().__init__(name=name)
        self._dim = dim
        self._dim_out = dim_out
        self._kernel_size = kernel_size
        self._stride = stride
        self._layer_norm_axis = layer_norm_axis

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        conv = hk.Conv1DTranspose(
            output_channels=self._dim if self._dim_out is None else self._dim_out,
            kernel_shape=self._kernel_size,
            padding="SAME",
            data_format="NWC",
            stride=self._stride,
        )

        layer_norm = hk.LayerNorm(
            axis=self._layer_norm_axis,
            create_scale=True,
            create_offset=True,
            eps=1e-5,
            param_axis=self._layer_norm_axis,
        )

        x = layer_norm(x)
        x = conv(x)
        x = jax.nn.gelu(x)
        return x
