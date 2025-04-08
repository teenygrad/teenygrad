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

use super::error::ViewError;

pub struct View {
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub mask: Option<Vec<(i64, i64)>>,
    pub contiguous: bool,
    pub offset: i64,
}

impl View {
    pub fn create(
        shape: Vec<i64>,
        strides: Vec<i64>,
        offset: i64,
        mask: Option<Vec<(i64, i64)>>,
        contiguous: bool,
    ) -> Result<Self, ViewError> {
        Ok(View {
            shape,
            strides,
            mask,
            contiguous,
            offset,
        })
    }
}

pub struct ViewBuilder {
    shape: Vec<i64>,
    strides: Vec<i64>,
    offset: i64,
    mask: Option<Vec<(i64, i64)>>,
}

impl ViewBuilder {
    pub fn new() -> Self {
        Self {
            shape: vec![],
            strides: vec![],
            mask: None,
            offset: 0,
        }
    }

    pub fn with_shape(mut self, shape: Vec<i64>) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_strides(mut self, strides: Vec<i64>) -> Self {
        self.strides = strides;
        self
    }

    pub fn with_mask(mut self, mask: Vec<(i64, i64)>) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn with_offset(mut self, offset: i64) -> Self {
        self.offset = offset;
        self
    }

    pub fn build(self) -> Result<View, ViewError> {
        let strides = if !self.strides.is_empty() {
            filter_strides(&self.shape, &self.strides)
        } else {
            strides_for_shape(&self.shape)
        };

        let contiguous = self.offset == 0
            && self.mask.is_none()
            && self
                .strides
                .iter()
                .zip(strides_for_shape(&self.shape).iter())
                .all(|(s1, s2)| s1 == s2);

        View::create(self.shape, strides, self.offset, self.mask, contiguous)
    }
}

impl Default for ViewBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn filter_strides(shape: &[i64], strides: &[i64]) -> Vec<i64> {
    println!("filter_strides: {:?}, {:?}", shape, strides);
    strides
        .iter()
        .zip(shape.iter())
        .map(|(st, sh)| if *sh == 1 { 0 } else { *st })
        .collect()
}

fn strides_for_shape(shape: &[i64]) -> Vec<i64> {
    let mut strides = if shape.is_empty() { vec![] } else { vec![1] };
    for d in shape.iter().skip(1).rev() {
        strides.insert(0, d * strides[0]);
    }
    println!("strides_for_shape-*1: {:?}", strides);
    filter_strides(shape, &strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_strides() {
        assert_eq!(filter_strides(&[], &[]), vec![], "Test empty arrays");

        assert_eq!(filter_strides(&[5], &[1]), vec![1], "Test single dimension");
        assert_eq!(
            filter_strides(&[1], &[10]),
            vec![0],
            "Test single dimension"
        );

        assert_eq!(
            filter_strides(&[3, 4], &[4, 1]),
            vec![4, 1],
            "Test multiple dimensions"
        );
        assert_eq!(
            filter_strides(&[1, 4], &[10, 1]),
            vec![0, 1],
            "Test multiple dimensions"
        );
        assert_eq!(
            filter_strides(&[4, 1], &[1, 10]),
            vec![1, 0],
            "Test multiple dimensions"
        );

        assert_eq!(
            filter_strides(&[1, 1, 1], &[10, 20, 30]),
            vec![0, 0, 0],
            "Test multiple size-1 dimensions"
        );

        assert_eq!(
            filter_strides(&[2, 1, 3, 1], &[12, 4, 1, 10]),
            vec![12, 0, 1, 0],
            "Test mixed dimensions"
        );
    }

    #[test]
    fn test_strides_for_shape() {
        assert_eq!(strides_for_shape(&[]), vec![], "Test empty shape");

        assert_eq!(strides_for_shape(&[5]), vec![1], "Test single dimension");

        assert_eq!(
            strides_for_shape(&[3, 4]),
            vec![4, 1],
            "Test two dimensions"
        );

        assert_eq!(
            strides_for_shape(&[2, 3, 4]),
            vec![12, 4, 1],
            "Test three dimensions"
        );

        assert_eq!(
            strides_for_shape(&[2, 3, 4, 5]),
            vec![60, 20, 5, 1],
            "Test four dimensions"
        );

        assert_eq!(
            strides_for_shape(&[1, 2, 1, 3]),
            vec![0, 3, 0, 1],
            "Test with size 1 dimensions"
        );

        assert_eq!(
            strides_for_shape(&[1, 1, 1]),
            vec![0, 0, 0],
            "Test with all size 1 dimensions"
        );

        assert_eq!(
            strides_for_shape(&[2, 1, 3, 1]),
            vec![3, 0, 1, 0],
            "Test with mixed dimensions including size 1"
        );
    }

    #[test]
    fn test_view_create() {
        let v = ViewBuilder::new().with_shape(vec![2, 3]).build().unwrap();
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(v.strides, vec![3, 1]);
        assert_eq!(v.offset, 0);
        assert!(v.mask.is_none());
        assert!(v.contiguous);
    }

    #[test]
    fn test_view_with_custom_strides() {
        let v = ViewBuilder::new()
            .with_shape(vec![2, 3])
            .with_strides(vec![6, 2])
            .build()
            .unwrap();
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(v.strides, vec![6, 2]);
        assert_eq!(v.offset, 0);
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_offset() {
        let v = ViewBuilder::new()
            .with_shape(vec![2, 3])
            .with_offset(5)
            .build()
            .unwrap();
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(v.strides, vec![3, 1]);
        assert_eq!(v.offset, 5);
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_mask() {
        let mask = vec![(0, 1), (1, 2)];
        let v = ViewBuilder::new()
            .with_shape(vec![2, 3])
            .with_mask(mask.clone())
            .build()
            .unwrap();
        assert_eq!(v.shape, vec![2, 3]);
        assert_eq!(v.strides, vec![3, 1]);
        assert_eq!(v.offset, 0);
        assert_eq!(v.mask, Some(mask));
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_size_1_dimensions() {
        let v = ViewBuilder::new().with_shape(vec![1, 3]).build().unwrap();
        assert_eq!(v.shape, vec![1, 3], "shape");
        assert_eq!(v.strides, vec![0, 1], "strides");
        assert_eq!(v.offset, 0, "offset");
        assert!(v.mask.is_none(), "mask");
        assert!(v.contiguous, "contiguous");
    }

    #[test]
    fn test_view_with_custom_strides_and_size_1_dimensions() {
        let v = ViewBuilder::new()
            .with_shape(vec![1, 3])
            .with_strides(vec![10, 2])
            .build()
            .unwrap();
        assert_eq!(v.shape, vec![1, 3]);
        assert_eq!(v.strides, vec![0, 2]);
        assert_eq!(v.offset, 0);
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_empty_shape() {
        let v = ViewBuilder::new().with_shape(vec![]).build().unwrap();
        assert_eq!(v.shape, vec![]);
        assert_eq!(v.strides, vec![]);
        assert_eq!(v.offset, 0);
        assert!(v.mask.is_none());
        assert!(v.contiguous);
    }
}
