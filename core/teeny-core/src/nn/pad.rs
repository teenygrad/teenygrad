/*
 * Copyright (c) 2026 Teenygrad.
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

macro_rules! pad_layer {
    (
        $name:ident,
        $( $field:ident : $fty:ty ),+
        $(,)?
    ) => {
        pub struct $name<D: Dtype, IT, OT, const RANK: usize> {
            $( pub $field: $fty, )+
            _pd: PhantomData<(D, IT, OT)>,
        }

        impl<D: Dtype, IT, OT, const RANK: usize> $name<D, IT, OT, RANK> {
            pub fn new( $( $field: $fty ),+ ) -> Self {
                Self { $( $field, )+ _pd: PhantomData }
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

// ConstantPad — adds a constant fill value on each side
pad_layer!(ConstantPad1d, pad_left: usize, pad_right: usize, value: f64);
pad_layer!(ConstantPad2d, pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize, value: f64);
pad_layer!(ConstantPad3d, pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize, value: f64);

// ReflectionPad — pads by reflecting the input at the boundary
pad_layer!(ReflectionPad1d, pad_left: usize, pad_right: usize);
pad_layer!(ReflectionPad2d, pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize);
pad_layer!(ReflectionPad3d, pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize);

// ReplicationPad — pads by replicating the edge values
pad_layer!(ReplicationPad1d, pad_left: usize, pad_right: usize);
pad_layer!(ReplicationPad2d, pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize);
pad_layer!(ReplicationPad3d, pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize);

// CircularPad — pads by wrapping around (circular/periodic boundary)
pad_layer!(CircularPad1d, pad_left: usize, pad_right: usize);
pad_layer!(CircularPad2d, pad_l: usize, pad_r: usize, pad_t: usize, pad_b: usize);
pad_layer!(CircularPad3d, pad_d1: usize, pad_d2: usize, pad_h1: usize, pad_h2: usize, pad_w1: usize, pad_w2: usize);
