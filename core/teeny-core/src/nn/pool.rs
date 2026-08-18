/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use core::marker::PhantomData;

use crate::{
    dtype::{Dtype, EagerTensor, Tensor},
    nn::Layer,
};

// ---------------------------------------------------------------------------
// Macro — reduces boilerplate for the nine pool variants
// ---------------------------------------------------------------------------

macro_rules! pool_layer_1d {
    (#[doc = $doc:expr] $name:ident) => {
        #[doc = $doc]
        pub struct $name<D: Dtype, IT, OT, const RANK: usize> {
            /// The pooling window length.
            pub kernel_l: usize,
            /// The stride between pooling windows.
            pub stride: usize,
            _pd: PhantomData<(D, IT, OT)>,
        }

        impl<D: Dtype, IT, OT, const RANK: usize> $name<D, IT, OT, RANK> {
            /// Creates a new layer with the given kernel size and stride.
            pub fn new(kernel_size: usize, stride: usize) -> Self {
                Self {
                    kernel_l: kernel_size,
                    stride,
                    _pd: PhantomData,
                }
            }
        }

        impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
            Layer<IT> for $name<D, IT, OT, RANK>
        {
            type Output = OT;
            fn call(&self, _input: IT) -> Self::Output {
                todo!()
            }
        }
    };
}

macro_rules! pool_layer_2d {
    (#[doc = $doc:expr] $name:ident) => {
        #[doc = $doc]
        pub struct $name<D: Dtype, IT, OT, const RANK: usize> {
            /// Pooling window height.
            pub kernel_h: usize,
            /// Pooling window width.
            pub kernel_w: usize,
            /// Vertical stride between pooling windows.
            pub stride_h: usize,
            /// Horizontal stride between pooling windows.
            pub stride_w: usize,
            /// Vertical padding applied before pooling.
            pub padding_h: usize,
            /// Horizontal padding applied before pooling.
            pub padding_w: usize,
            _pd: PhantomData<(D, IT, OT)>,
        }

        impl<D: Dtype, IT, OT, const RANK: usize> $name<D, IT, OT, RANK> {
            /// Creates a new layer with the given kernel size and stride, no padding.
            pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
                Self {
                    kernel_h: kernel_size.0,
                    kernel_w: kernel_size.1,
                    stride_h: stride.0,
                    stride_w: stride.1,
                    padding_h: 0,
                    padding_w: 0,
                    _pd: PhantomData,
                }
            }

            /// Creates a new layer with the given kernel size, stride, and padding.
            pub fn with_padding(
                kernel_size: (usize, usize),
                stride: (usize, usize),
                padding: (usize, usize),
            ) -> Self {
                Self {
                    kernel_h: kernel_size.0,
                    kernel_w: kernel_size.1,
                    stride_h: stride.0,
                    stride_w: stride.1,
                    padding_h: padding.0,
                    padding_w: padding.1,
                    _pd: PhantomData,
                }
            }
        }

        impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
            Layer<IT> for $name<D, IT, OT, RANK>
        {
            type Output = OT;
            fn call(&self, _input: IT) -> Self::Output {
                todo!()
            }
        }
    };
}

macro_rules! pool_layer_3d {
    (#[doc = $doc:expr] $name:ident) => {
        #[doc = $doc]
        pub struct $name<D: Dtype, IT, OT, const RANK: usize> {
            /// Pooling window depth.
            pub kernel_d: usize,
            /// Pooling window height.
            pub kernel_h: usize,
            /// Pooling window width.
            pub kernel_w: usize,
            /// Stride along the depth dimension.
            pub stride_d: usize,
            /// Stride along the height dimension.
            pub stride_h: usize,
            /// Stride along the width dimension.
            pub stride_w: usize,
            _pd: PhantomData<(D, IT, OT)>,
        }

        impl<D: Dtype, IT, OT, const RANK: usize> $name<D, IT, OT, RANK> {
            /// Creates a new layer with the given kernel size and stride.
            pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
                Self {
                    kernel_d: kernel_size.0,
                    kernel_h: kernel_size.1,
                    kernel_w: kernel_size.2,
                    stride_d: stride.0,
                    stride_h: stride.1,
                    stride_w: stride.2,
                    _pd: PhantomData,
                }
            }
        }

        impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize>
            Layer<IT> for $name<D, IT, OT, RANK>
        {
            type Output = OT;
            fn call(&self, _input: IT) -> Self::Output {
                todo!()
            }
        }
    };
}

// ---------------------------------------------------------------------------
// AvgPool
// ---------------------------------------------------------------------------

pool_layer_1d!(
    #[doc = "1-D average pooling."]
    AvgPool1d
);
pool_layer_2d!(
    #[doc = "2-D average pooling."]
    AvgPool2d
);
pool_layer_3d!(
    #[doc = "3-D average pooling."]
    AvgPool3d
);

// ---------------------------------------------------------------------------
// MaxPool
// ---------------------------------------------------------------------------

pool_layer_1d!(
    #[doc = "1-D max pooling."]
    MaxPool1d
);
pool_layer_2d!(
    #[doc = "2-D max pooling."]
    MaxPool2d
);
pool_layer_3d!(
    #[doc = "3-D max pooling."]
    MaxPool3d
);

// ---------------------------------------------------------------------------
// LpPool — like AvgPool but with a configurable p-norm
// ---------------------------------------------------------------------------

/// 1-D power-average (Lp) pooling: like average pooling, generalized to an arbitrary p-norm.
pub struct LpPool1d<D: Dtype, IT, OT, const RANK: usize> {
    /// The pooling window length.
    pub kernel_l: usize,
    /// The stride between pooling windows.
    pub stride: usize,
    /// The `p` in the p-norm.
    pub p: f64,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> LpPool1d<D, IT, OT, RANK> {
    /// Creates a new layer with the given kernel size, stride, and `p`-norm.
    pub fn new(kernel_size: usize, stride: usize, p: f64) -> Self {
        Self {
            kernel_l: kernel_size,
            stride,
            p,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for LpPool1d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

/// 2-D power-average (Lp) pooling: like average pooling, generalized to an arbitrary p-norm.
pub struct LpPool2d<D: Dtype, IT, OT, const RANK: usize> {
    /// Pooling window height.
    pub kernel_h: usize,
    /// Pooling window width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// The `p` in the p-norm.
    pub p: f64,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> LpPool2d<D, IT, OT, RANK> {
    /// Creates a new layer with the given kernel size, stride, and `p`-norm.
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), p: f64) -> Self {
        Self {
            kernel_h: kernel_size.0,
            kernel_w: kernel_size.1,
            stride_h: stride.0,
            stride_w: stride.1,
            p,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for LpPool2d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}

/// 3-D power-average (Lp) pooling: like average pooling, generalized to an arbitrary p-norm.
pub struct LpPool3d<D: Dtype, IT, OT, const RANK: usize> {
    /// Pooling window depth.
    pub kernel_d: usize,
    /// Pooling window height.
    pub kernel_h: usize,
    /// Pooling window width.
    pub kernel_w: usize,
    /// Stride along the depth dimension.
    pub stride_d: usize,
    /// Stride along the height dimension.
    pub stride_h: usize,
    /// Stride along the width dimension.
    pub stride_w: usize,
    /// The `p` in the p-norm.
    pub p: f64,
    _pd: PhantomData<(D, IT, OT)>,
}

impl<D: Dtype, IT, OT, const RANK: usize> LpPool3d<D, IT, OT, RANK> {
    /// Creates a new layer with the given kernel size, stride, and `p`-norm.
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize), p: f64) -> Self {
        Self {
            kernel_d: kernel_size.0,
            kernel_h: kernel_size.1,
            kernel_w: kernel_size.2,
            stride_d: stride.0,
            stride_h: stride.1,
            stride_w: stride.2,
            p,
            _pd: PhantomData,
        }
    }
}

impl<D: Dtype, IT: Tensor<D, RANK> + EagerTensor, OT: Tensor<D, RANK>, const RANK: usize> Layer<IT>
    for LpPool3d<D, IT, OT, RANK>
{
    type Output = OT;
    fn call(&self, _input: IT) -> Self::Output {
        todo!()
    }
}
