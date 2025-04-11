/*
 * Copyright (c) SpinorML 2025. All rights reserved.
 *
 * This software and associated documentation files (the "Software") are proprietary
 * and confidential. The Software is protected by copyright laws and international
 * copyright treaties, as well as other intellectual property laws and treaties.
 *
 * No part of this Software may be reproduced, distributed, or transmitted in any
 * form or by any means, including photocopying, recording, or other electronic or
 * mechanical methods, without the prior written permission of SpinorML.
 *
 * Unauthorized copying, modification, distribution, or use of this Software is
 * strictly prohibited and may result in severe civil and criminal penalties.
 */

use alloc::vec::Vec;

use crate::tensor::{ElementType, Tensor};

pub struct MemoryTensor<T> {
    pub element_type: ElementType,
    pub shape: Vec<i64>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> for MemoryTensor<T> {
    fn element_type(&self) -> &ElementType {
        &self.element_type
    }

    fn shape(&self) -> &[i64] {
        &self.shape
    }

    fn data(&self) -> &[T] {
        &self.data
    }

    fn reshape(&mut self, shape: Vec<i64>) -> &mut Self {
        self.shape = shape;
        self
    }
}

impl<T> MemoryTensor<T> {
    pub fn new(element_type: ElementType, shape: Vec<i64>, data: Vec<T>) -> Self {
        Self {
            element_type,
            shape,
            data,
        }
    }
}

impl<T> Default for MemoryTensor<T> {
    fn default() -> Self {
        Self::new(ElementType::FP16, Vec::new(), Vec::new())
    }
}
