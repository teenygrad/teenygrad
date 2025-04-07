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

use super::error::ShapeError;

pub struct View {
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub mask: Option<Vec<(i64, i64)>>,
    pub contiguous: bool,
    pub offset: i64,
}

impl View {
    pub fn create(
        shape: &[i64],
        strides: &[i64],
        mask: Option<&[(i64, i64)]>,
    ) -> Result<Self, ShapeError> {
        if !shape.iter().all(|s| *s >= 0) {
            return Err(ShapeError::NegativeDimension());
        }

        let strides = if strides.is_empty() {
            Self::strides_for_shape(shape)
        } else {
            Self::canonicalize_strides(shape, strides)
        };

        if shape.iter().any(|s| *s == 0) {
            return Ok(View {
                shape: shape.to_vec(),
                strides: vec![0; shape.len()],
                mask: None,
                contiguous: true,
                offset: 0,
            });
        }

        if (mask.is_some() && shape.iter().zip(mask.unwrap()).all(|(s, m)| m == (0, s))) {
            todo!()
        }
        // # canonicalize no-op mask
        // if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
        // # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
        // # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
        // if mask and any(elim := [not resolve(b+1 < e) for b,e in mask]):
        //   if any(not resolve(b < e) for b,e in mask):
        //     strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
        //   offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
        //   strides = tuple(0 if e else st for st,e in zip(strides, elim))
        // # simplify as we go
        // if isinstance(offset, UOp): offset = cast(sint, offset.ssimplify())
        // shape = tuple(cast(sint, x.ssimplify()) if isinstance(x, UOp) else x for x in shape)
        // # TODO: enabling stride simplification breaks symbolic jit
        // """
        // strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
        // if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
        // """
        // contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
        // return View(shape, strides, offset, mask, contiguous)
        Ok(Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            mask: mask.map(|m| m.to_vec()),
        })
    }

    fn filter_strides(shape: &[i64], strides: &[i64]) -> Vec<i64> {
        shape
            .iter()
            .zip(strides.iter())
            .map(|(s, st)| if *s == 1 { 0 } else { *st })
            .collect()
    }

    fn canonicalize_strides(shape: &[i64], strides: &[i64]) -> Vec<i64> {
        let mut strides = strides.to_vec();
        for (i, s) in shape.iter().enumerate() {
            if *s == 1 {
                strides[i] = 0;
            }
        }
        strides
    }

    fn strides_for_shape(shape: &[i64]) -> Vec<i64> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_strides() {
        // Test empty arrays
        assert_eq!(View::filter_strides(&[], &[]), vec![]);

        // Test single dimension
        assert_eq!(View::filter_strides(&[5], &[1]), vec![1]);
        assert_eq!(View::filter_strides(&[1], &[10]), vec![0]);

        // Test multiple dimensions
        assert_eq!(View::filter_strides(&[3, 4], &[4, 1]), vec![4, 1]);
        assert_eq!(View::filter_strides(&[1, 4], &[10, 1]), vec![0, 1]);
        assert_eq!(View::filter_strides(&[4, 1], &[1, 10]), vec![1, 0]);

        // Test multiple size-1 dimensions
        assert_eq!(
            View::filter_strides(&[1, 1, 1], &[10, 20, 30]),
            vec![0, 0, 0]
        );

        // Test mixed dimensions
        assert_eq!(
            View::filter_strides(&[2, 1, 3, 1], &[12, 4, 1, 10]),
            vec![12, 0, 1, 0]
        );
    }

    #[test]
    fn test_canonicalize_strides() {
        // Test empty arrays
        assert_eq!(View::canonicalize_strides(&[], &[]), vec![]);

        // Test single dimension
        assert_eq!(View::canonicalize_strides(&[5], &[1]), vec![1]);
        assert_eq!(View::canonicalize_strides(&[1], &[10]), vec![0]);

        // Test multiple dimensions
        assert_eq!(View::canonicalize_strides(&[3, 4], &[4, 1]), vec![4, 1]);
        assert_eq!(View::canonicalize_strides(&[1, 4], &[10, 1]), vec![0, 1]);
        assert_eq!(View::canonicalize_strides(&[4, 1], &[1, 10]), vec![1, 0]);

        // Test multiple size-1 dimensions
        assert_eq!(
            View::canonicalize_strides(&[1, 1, 1], &[10, 20, 30]),
            vec![0, 0, 0]
        );

        // Test mixed dimensions
        assert_eq!(
            View::canonicalize_strides(&[2, 1, 3, 1], &[12, 4, 1, 10]),
            vec![12, 0, 1, 0]
        );
    }

    #[test]
    fn test_strides_for_shape() {
        assert_eq!(View::strides_for_shape(&[]), vec![], "Test empty shape");
        assert_eq!(
            View::strides_for_shape(&[5]),
            vec![1],
            "Test single dimension"
        );
        assert_eq!(
            View::strides_for_shape(&[3, 4]),
            vec![4, 1],
            "Test two dimensions"
        );
        assert_eq!(
            View::strides_for_shape(&[2, 3, 4]),
            vec![12, 4, 1],
            "Test three dimensions"
        );
        assert_eq!(
            View::strides_for_shape(&[2, 3, 4, 5]),
            vec![60, 20, 5, 1],
            "Test four dimensions"
        );
        assert_eq!(
            View::strides_for_shape(&[1, 2, 1, 3]),
            vec![6, 3, 3, 1],
            "Test with size 1 dimensions"
        );
    }

    // #[test]
    // fn test_canonicalize_empty_mask() {
    //     let v = View::create(&[2, 2, 2], &[4, 2, 1], Some(&[(0, 2), (0, 2), (0, 2)]));
    //     assert!(v.mask.is_none());
    //     let v = View::create(&[4, 3, 2], &[1, 4, 10], Some(&[(0, 4), (0, 3), (0, 2)]));
    //     assert!(v.mask.is_none());
    // }

    //     #!/usr/bin/env python
    // import unittest
    // from tinygrad.shape.view import View, merge_dims
    // # from tinygrad.shape.shapetracker import ShapeTracker

    // class TestView(unittest.TestCase):
    //   def test_canonicalize_empty_mask(self):
    //     v = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    //     self.assertIsNone(v.mask)
    //     v = View.create(shape=(4,3,2), strides=(1,4,10), mask=((0,4),(0,3),(0,2)))
    //     self.assertIsNone(v.mask)

    //   def test_minify_zero_strided_dims(self):
    //     target = View.create(shape=(2,2), strides=(30,2), offset=7, mask=None)
    //     v = View.create(shape=(2,1,2), strides=(30,0,2), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(1,2,2), strides=(0,30,2), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(2,2,1), strides=(30,2,0), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(2,1,1,2), strides=(30,0,0,2), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(1,1,2,2), strides=(0,0,30,2), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(2,2,1,1), strides=(30,2,0,0), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(1,2,2,1), strides=(0,30,2,0), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)
    //     v = View.create(shape=(1,2,1,2), strides=(0,30,0,2), offset=7, mask=None)
    //     self.assertEqual(v.minify(), target)

    //   def test_empty_mask_contiguous(self):
    //     v1 = View.create(shape=(2,2,2), strides=(4,2,1), mask=None)
    //     v2 = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    //     self.assertEqual(v1.contiguous, v2.contiguous)
    //     v1 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=None)
    //     v2 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=((0,1),(0,1),(0,1),(0,4)))
    //     self.assertEqual(v1.contiguous, v2.contiguous)
    //     v = View.create(shape=(2,3,4), mask=((0,2),(0,3),(0,4)))
    //     self.assertTrue(v.contiguous)

    //   def test_reshape_all_invalid(self):
    //     v = View.create((4,5), mask=((0,0), (0,0))).reshape((20,))
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View.create((20,), mask=((0,0),)))

    // class TestMergeDims(unittest.TestCase):
    //   def test_contiguous(self):
    //     shape = (2, 3, 4)
    //     strides = (12, 4, 1) #=strides_for_shape(shape)
    //     m = merge_dims(shape, strides)
    //     self.assertEqual(m, ((24, 1, 24),))

    //   def test_0_in_strides(self):
    //     shape = (2, 3, 4)
    //     self.assertEqual(merge_dims(shape, (0, 4, 1)), ((2, 0, 0), (12, 1, 12)))
    //     self.assertEqual(merge_dims(shape, (0, 0, 1)), ((6, 0, 0), (4, 1, 4)))
    //     self.assertEqual(merge_dims(shape, (3, 1, 0)), ((6, 1, 6), (4, 0, 4)))
    //     self.assertEqual(merge_dims(shape, (0, 0, 0)), ((24, 0, 0),))

    //   def test_pad(self):
    //     # print(ShapeTracker.from_shape((1, 2)).pad(((1, 0), (0, 1))).views[-1])
    //     self.assertEqual(merge_dims((2, 3), (0, 1), ((1, 2), (0, 2))), ((6, 1, 3),))

    //     # print(f"{ShapeTracker.from_shape((1, 1, 2)).pad(((1, 0), (1, 0), (0, 1))).views[-1]}")
    //     self.assertEqual(merge_dims((2, 2, 3), (0, 0, 1), ((1, 2), (1, 2), (0, 2))), ((12, 1, 3),))

    //     # print(f"{ShapeTracker.from_shape((1, 1, 2, 2)).pad(((1, 0), (1, 0), (0, 1), (0, 1))).views[-1]}")
    //     self.assertEqual(merge_dims((2, 2, 3, 3), (0, 0, 2, 1), ((1, 2), (1, 2), (0, 2), (0, 2))), ((12, 2, 3), (3, 1, 3)))

    //     # print(f"{ShapeTracker.from_shape((2, 1, 2)).pad(((0, 0), (1, 0), (0, 1))).views[-1]}")
    //     self.assertEqual(merge_dims((2, 2, 3), (2, 0, 1), ((0, 2), (1, 2), (0, 2))), ((2, 2, 2), (6, 1, 3)))

    //   def test_different_1_pad(self):
    //     # print(f"{ShapeTracker.from_shape((2, 2, 1)).pad(((0, 0), (0, 0), (0, 1))).views[-1]}")
    //     self.assertEqual(merge_dims((2, 2, 2), (2, 1, 0), ((0, 2), (0, 2), (0, 1))), ((4, 1, 4), (2, 0, 2)))

    //     # print(f"{ShapeTracker.from_shape((2, 1, 1)).pad(((0, 0), (0, 1), (0, 1))).views[-1]}")
    //     self.assertEqual(merge_dims((2, 2, 2), (1, 0, 0), ((0, 2), (0, 2), (0, 1))), ((2, 1, 2), (4, 0, 4)))

    // class TestMergeViews(unittest.TestCase):
    //   def test_with_mask_0(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     v0 = View(shape=(1, 1, 5, 8), strides=(0, 0, 5, 1), offset=-3, mask=((0, 1), (0, 1), (0, 5), (3, 8)), contiguous=False)
    //     v1 = View(shape=(1, 1, 2, 2), strides=(0, 0, 8, 1), offset=3, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(1, 1, 2, 2), strides=(0, 0, 5, 1), offset=0, mask=None, contiguous=False))

    //   def test_with_mask_1(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     v0 = View(shape=(3, 3, 5, 3), strides=(27, 9, 3, 1), offset=-6, mask=((0, 3), (0, 3), (2, 4), (1, 3)), contiguous=False)
    //     v1 = View(shape=(3, 3, 2, 2), strides=(45, 15, 3, 1), offset=7, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(3, 3, 2, 2), strides=(27, 9, 3, 1), offset=1, mask=None, contiguous=False))

    //   def test_with_mask_2(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     v0 = View(shape=(3, 3, 5, 3), strides=(27, 9, -3, 1), offset=6, mask=((0, 3), (0, 3), (0, 2), (0, 2)), contiguous=False)
    //     v1 = View(shape=(3, 3, 2, 2), strides=(45, 15, -3, 1), offset=3, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(3, 3, 2, 2), strides=(27, 9, 3, 1), offset=3, mask=None, contiguous=False))

    //   def test_with_mask_3(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     # has a mask in the final view
    //     v0 = View(shape=(3, 3, 4, 4), strides=(27, 9, 3, 1), offset=-5, mask=((0, 3), (0, 3), (2, 4), (0, 2)), contiguous=False)
    //     v1 = View(shape=(3, 3, 4, 2), strides=(48, 16, 4, 1), offset=0, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(3, 3, 4, 2), strides=(27, 9, 3, 1), offset=-5, mask=((0, 3), (0, 3), (2, 4), (0, 2)), contiguous=False))

    //   def test_with_mask_4(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     # has a mask in the final view
    //     v0 = View(shape=(3, 3, 5, 3), strides=(27, 9, -3, 1), offset=6, mask=((0, 3), (0, 3), (0, 2), (1, 3)), contiguous=False)
    //     v1 = View(shape=(3, 3, 3, 3), strides=(45, 15, 3, 1), offset=6, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(3, 3, 3, 3), strides=(0, 0, 0, 0), offset=0, mask=((0, 0), (0, 0), (0, 0), (0, 0)), contiguous=False))

    //   def test_with_mask_5(self):
    //     # from test/test_ops.py::TestOps::test_pad_reflect_mode
    //     # has a mask in the final view
    //     v0 = View(shape=(1, 1, 6, 5), strides=(0, 0, 5, 1), offset=-5, mask=((0, 1), (0, 1), (1, 6), (0, 5)), contiguous=False)
    //     v1 = View(shape=(1, 1, 6, 3), strides=(0, 0, 5, -1), offset=3, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(1, 1, 6, 3), strides=(0, 0, 5, -1), offset=-2, mask=((0, 1), (0, 1), (1, 6), (0, 3)), contiguous=False))

    //   @unittest.expectedFailure  # TODO: fix these
    //   def test_merges_from_fuzzer1(self):
    //     v0 = View(shape=(2, 4), strides=(2, 1), offset=-2, mask=((0, 2), (2, 4)), contiguous=False)
    //     v1 = View(shape=(2, 4, 2, 2), strides=(4, 0, -2, -1), offset=3, mask=None, contiguous=False)
    //     target = View(shape=(2, 4, 2, 2), strides=(2, 0, 0, -1), offset=1, mask=((0, 2), (0, 4), (0, 1), (0, 2)), contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, target)

    //   @unittest.expectedFailure  # TODO: fix these
    //   def test_merges_from_fuzzer2(self):
    //     v0 = View(shape=(5, 10, 12), strides=(100, 1, 10), offset=-20, mask=((0, 5), (0, 10), (2, 12)), contiguous=False)
    //     v1 = View(shape=(10, 6, 5, 2, 2), strides=(12, 2, 120, 1, 0), offset=0, mask=None, contiguous=False)
    //     target = View(shape=(10, 6, 5, 2, 2), strides=(1, 20, 100, 10, 0), offset=-20, mask=((0, 10), (1, 6), (0, 5), (0, 2), (0, 2)), contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, target)

    //   @unittest.expectedFailure  # TODO: fix these
    //   def test_merges_from_fuzzer3(self):
    //     v0 = View(shape=(8, 7, 3), strides=(1, 12, -4), offset=6, mask=((2, 6), (0, 7), (0, 3)), contiguous=False)
    //     v1 = View(shape=(4, 2, 6, 2, 1), strides=(42, 21, 3, 1, 0), offset=4, mask=None, contiguous=False)
    //     target = View(shape=(4, 2, 6, 2, 1), strides=(2, 1, 12, -4, 0), offset=14, mask=((1, 3), (0, 2), (0, 6), (0, 2), (0, 1)), contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, target)

    //   @unittest.expectedFailure  # TODO: fix these
    //   def test_merges_from_fuzzer4(self):
    //     v0 = View(shape=(7, 21, 3), strides=(54, 3, 1), offset=-9, mask=((0, 6), (3, 21), (0, 3)), contiguous=False)
    //     v1 = View(shape=(5, 3, 3, 7), strides=(63, 1, 3, 9), offset=63, mask=None, contiguous=False)
    //     target = View(shape=(5, 3, 3, 7), strides=(54, 1, 3, 9), offset=45, mask=((0, 5), (0, 3), (0, 3), (1, 7)), contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, target)

    //   @unittest.expectedFailure  # TODO: fix these
    //   def test_merges_from_fuzzer5(self):
    //     v0 = View(shape=(5, 1, 24), strides=(20, 0, 1), offset=-2, mask=((0, 5), (0, 1), (2, 22)), contiguous=False)
    //     v1 = View(shape=(12, 2, 5, 2, 1), strides=(2, 1, 24, 0, 0), offset=0, mask=None, contiguous=False)
    //     target = View(shape=(12, 2, 5, 2, 1), strides=(2, 1, 20, 0, 0), offset=-2, mask=((1, 11), (0, 2), (0, 5), (0, 2), (0, 1)), contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, target)

    //   def test_view_padded_area1(self):
    //     # test_multinomial
    //     v0 = View(shape=(2,), strides=(0,), offset=0, mask=((1, 2),), contiguous=False)
    //     v1 = View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(1,), strides=(0,), offset=0, mask=((0, 0),), contiguous=False))

    //   def test_view_padded_area2(self):
    //     # test_pad_reflect_mode
    //     v0 = View(shape=(1, 1, 10, 7), strides=(0, 0, 5, 1), offset=-15, mask=((0, 1), (0, 1), (3, 8), (0, 5)), contiguous=False)
    //     v1 = View(shape=(0, 0, 0, 0), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=True)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(0, 0, 0, 0), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=True))

    //   def test_view_padded_area3(self):
    //     # test_roll
    //     v0 = View(shape=(2, 4), strides=(0, 1), offset=4, mask=((0, 1), (0, 4)), contiguous=False)
    //     v1 = View(shape=(1, 4), strides=(0, 1), offset=4, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(1, 4), strides=(0, 0), offset=0, mask=((0, 0), (0, 0)), contiguous=False))

    //   def test_view_padded_area4(self):
    //     # test_std_mean
    //     v0 = View(shape=(2,), strides=(0,), offset=0, mask=((0, 1),), contiguous=False)
    //     v1 = View(shape=(1, 1, 1), strides=(0, 0, 0), offset=1, mask=None, contiguous=False)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=((0, 0), (0, 0), (0, 0)), contiguous=False))

    //   def test_empty_shape_view1(self):
    //     # test_stack_slice
    //     v0 = View(shape=(3, 5), strides=(0, 1), offset=0, mask=((0, 1), (0, 5)), contiguous=False)
    //     v1 = View(shape=(), strides=(), offset=0, mask=None, contiguous=True)
    //     v = v0 + v1
    //     self.assertIsNotNone(v)
    //     self.assertEqual(v, View(shape=(), strides=(), offset=0, mask=None, contiguous=True))

    //   def test_empty_shape_view2(self):
    //     # test_std_mean
    //     v0 = View(shape=(2,), strides=(0,), offset=0, mask=((1, 2),), contiguous=False)
    //     v1 = View(shape=(), strides=(), offset=0, mask=None, contiguous=True)
    //     v = v0 + v1
    //     # TODO: why is this different?
    //     self.assertIsNone(v)

    // if __name__ == '__main__':
    //   unittest.main()
}
