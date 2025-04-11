/*
 * Copyright (C) 2025 SpinorML Ltd.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use super::error::ViewError;
use super::symbolic::NodeOrInt;
use super::symbolic::var_node::VarNode;
use alloc::vec::Vec;
pub struct View {
    pub shape: Vec<NodeOrInt>,
    pub strides: Vec<NodeOrInt>,
    pub mask: Option<Vec<(NodeOrInt, NodeOrInt)>>,
    pub contiguous: bool,
    pub offset: NodeOrInt,
}

impl View {
    pub fn create(
        shape: Vec<NodeOrInt>,
        strides: Vec<NodeOrInt>,
        offset: NodeOrInt,
        mask: Option<Vec<(NodeOrInt, NodeOrInt)>>,
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

    //   def size(self): return prod([s.max if isinstance(s, Node) else s for s,st in zip(self.shape, self.strides) if st != 0])
    pub fn size(&self) -> i64 {
        todo!()
    }

    //   def vars(self) -> List[Variable]:
    //     flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
    //     return dedup(functools.reduce(operator.add, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, Node)], []))
    pub fn vars(&self) -> Vec<VarNode> {
        let _flatten_mask = if let Some(ref mask) = self.mask {
            mask.iter()
                .flat_map(|m| Vec::from([m.0.clone(), m.1.clone()]))
                .collect()
        } else {
            Vec::new()
        };

        todo!()
    }

    //   def unbind(self) -> View:
    //     unbound_vars:Dict[VariableOrNum,Node] = {v: v.unbind()[0] for v in self.vars() if v.val is not None}
    //     new_shape = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.shape])
    //     new_strides = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.strides])
    //     new_offset = self.offset if isinstance(self.offset, int) else self.offset.substitute(unbound_vars)
    //     new_mask = tuple((a if isinstance(a, int) else a.substitute(unbound_vars), b if isinstance(b, int) else b.substitute(unbound_vars)) for (a, b) in self.mask) if self.mask is not None else None
    //     return View.create(new_shape, new_strides, new_offset, new_mask)
    pub fn unbind(&self) -> View {
        todo!()
    }

    //   # MovementOps live here now

    //   def __unsafe_resize(self, arg: Tuple[Tuple[sint, sint], ...], mask=None) -> View:
    //     offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
    //     if self.mask:
    //       # move the old mask
    //       nmask = tuple([(max(mx-ax, 0), min(my-ax, ay-ax)) for (mx,my),(ax,ay) in zip(self.mask, arg)])
    //       # merge the masks if we have two
    //       mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    //     shape = [y-x for x,y in arg]
    //     return View.create(tuple(s.b if isinstance(s, NumNode) else s for s in shape), self.strides, self.offset+offset, mask)
    pub fn __unsafe_resize(&self, _arg: Vec<(i64, i64)>, _mask: Option<Vec<(i64, i64)>>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def pad(self, arg: Tuple[Tuple[int, int], ...]) -> View:
    //     assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape)
    //     if any(b or e for b, e in arg):
    //       zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
    //       mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
    //       return self.__unsafe_resize(zvarg, mask=mask)
    //     return self
    pub fn pad(&self, _arg: Vec<(i64, i64)>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
    //     assert all((b>=0 and e<=s) for s,(b,e) in zip(self.shape,arg)) and len(arg) == len(self.shape)
    //     return self.__unsafe_resize(arg)
    pub fn shrink(&self, _arg: Vec<(i64, i64)>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def expand(self, new_shape: Tuple[sint, ...]) -> View:
    //     assert len(new_shape) == len(self.shape)
    //     assert all(is_sym_int(x) and (s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.strides)), f"can't expand {self.shape} into {new_shape}"
    //     # NOTE: can the mask ever be (0,0)?
    //     mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
    //     return View.create(new_shape, self.strides, self.offset, mask)
    pub fn expand(&self, _new_shape: Vec<i64>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def permute(self, axis: Tuple[int, ...]) -> View:
    //     assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    //     assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    //     return View.create(tuple([self.shape[a] for a in axis]), tuple([self.strides[a] for a in axis]), self.offset, tuple([self.mask[a] for a in axis]) if self.mask is not None else None)
    pub fn permute(&self, _axis: Vec<i64>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def stride(self, mul: Tuple[int, ...]) -> View:
    //     # except for the negative case, you can build this from the others. invertible in the negative case
    //     assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    //     strides = tuple([z*m for z,m in zip(self.strides, mul)])
    //     new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)])
    //     offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    //     mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) for (mx,my),s,m in zip(self.mask, self.shape, mul)]) if self.mask is not None else None
    //     return View.create(new_shape, strides, self.offset + offset, mask)
    pub fn stride(&self, _mul: Vec<i64>) -> View {
        todo!()
    }

    //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    //   def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
    //     if self.shape == new_shape: return self
    //     assert all(is_sym_int(x) and x > 0 for x in new_shape), f"shape must be symbolic ints and can't contain 0 or negative numbers {new_shape}"
    //     # check for the same size
    //     if all_int(self.shape):
    //       if all_int(new_shape):
    //         assert prod(self.shape) == prod(new_shape), f"size mismatched, can't reshape {self.shape=} -> {new_shape=}"
    //       else:
    //         assert all(isinstance(s, (int, Variable)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"
    //         assert prod(self.shape) == prod([s if isinstance(s, int) else cast(Variable,s).val for s in new_shape]), f"size mismatched, can't reshape {self.shape=} -> {new_shape=}"

    //     # after the asserts, it's okay to check contiguous
    //     if self.contiguous: return View.create(new_shape)

    //     # check if this is adding or removing 1s (only)
    //     # NOTE: this is optional, but removes most calls to (expensive!) merge_views (with mask, not optional)
    //     if [x for x in self.shape if x != 1] == [x for x in new_shape if x != 1]:
    //       new_strides: List[sint] = [y for x,y in zip(self.shape, self.strides) if x != 1]
    //       new_strides_tuple: Tuple[sint, ...] = tuple([0 if x == 1 else new_strides.pop(0) for x in new_shape])
    //       new_mask_tuple: Optional[Tuple[Tuple[sint, sint], ...]] = None
    //       if self.mask:
    //         for x,y in zip(self.shape, self.mask):
    //           if x == 1 and y != (0, 1):
    //             new_mask_tuple = ((0,0),) * len(new_shape)
    //             break
    //         else:
    //           new_mask: List[Tuple[sint, sint]] = [y for x,y in zip(self.shape, self.mask) if x != 1]
    //           new_mask_tuple = tuple([(0,1) if x == 1 else new_mask.pop(0) for x in new_shape])
    //       return View.create(new_shape, new_strides_tuple, self.offset, new_mask_tuple)

    //     # TODO: bring the merge_views logic here for more caching

    //     return None
    pub fn reshape(&self, _new_shape: Vec<i64>) -> Option<View> {
        todo!()
    }
}

pub struct ViewBuilder {
    shape: Vec<NodeOrInt>,
    strides: Vec<NodeOrInt>,
    offset: NodeOrInt,
    mask: Option<Vec<(NodeOrInt, NodeOrInt)>>,
}

impl ViewBuilder {
    pub fn new() -> Self {
        Self {
            shape: Vec::new(),
            strides: Vec::new(),
            mask: None,
            offset: NodeOrInt::Int(0),
        }
    }

    pub fn with_shape(mut self, shape: Vec<NodeOrInt>) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_strides(mut self, strides: Vec<NodeOrInt>) -> Self {
        self.strides = strides;
        self
    }

    pub fn with_mask(mut self, mask: Vec<(NodeOrInt, NodeOrInt)>) -> Self {
        self.mask = Some(mask);
        self
    }

    pub fn with_offset(mut self, offset: NodeOrInt) -> Self {
        self.offset = offset;
        self
    }

    pub fn build(self) -> Result<View, ViewError> {
        let strides = if !self.strides.is_empty() {
            filter_strides(&self.shape, &self.strides)
        } else {
            strides_for_shape(&self.shape)
        };

        let contiguous = self.offset == NodeOrInt::Int(0)
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

fn filter_strides(shape: &[NodeOrInt], strides: &[NodeOrInt]) -> Vec<NodeOrInt> {
    strides
        .iter()
        .zip(shape.iter())
        .map(|(st, sh)| {
            if *sh == NodeOrInt::Int(1) {
                NodeOrInt::Int(0)
            } else {
                st.clone()
            }
        })
        .collect()
}

fn strides_for_shape(shape: &[NodeOrInt]) -> Vec<NodeOrInt> {
    let strides = if shape.is_empty() {
        Vec::new()
    } else {
        Vec::from([NodeOrInt::Int(1)])
    };
    for _d in shape.iter().skip(1).rev() {
        todo!()
        //strides.insert(0, d * strides[0]);
    }
    filter_strides(shape, &strides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_strides() {
        assert_eq!(filter_strides(&[], &[]), Vec::new(), "Test empty arrays");

        assert_eq!(
            filter_strides(&[NodeOrInt::Int(5)], &[NodeOrInt::Int(1)]),
            Vec::from([NodeOrInt::Int(1)]),
            "Test single dimension"
        );
        assert_eq!(
            filter_strides(&[NodeOrInt::Int(1)], &[NodeOrInt::Int(10)]),
            Vec::from([NodeOrInt::Int(0)]),
            "Test single dimension"
        );

        assert_eq!(
            filter_strides(
                &[NodeOrInt::Int(3), NodeOrInt::Int(4)],
                &[NodeOrInt::Int(4), NodeOrInt::Int(1)]
            ),
            Vec::from([NodeOrInt::Int(4), NodeOrInt::Int(1)]),
            "Test multiple dimensions"
        );
        assert_eq!(
            filter_strides(
                &[NodeOrInt::Int(1), NodeOrInt::Int(4)],
                &[NodeOrInt::Int(10), NodeOrInt::Int(1)]
            ),
            Vec::from([NodeOrInt::Int(0), NodeOrInt::Int(1)]),
            "Test multiple dimensions"
        );
        assert_eq!(
            filter_strides(
                &[NodeOrInt::Int(4), NodeOrInt::Int(1)],
                &[NodeOrInt::Int(1), NodeOrInt::Int(10)]
            ),
            Vec::from([NodeOrInt::Int(1), NodeOrInt::Int(0)]),
            "Test multiple dimensions"
        );

        assert_eq!(
            filter_strides(
                &[NodeOrInt::Int(1), NodeOrInt::Int(1), NodeOrInt::Int(1)],
                &[NodeOrInt::Int(10), NodeOrInt::Int(20), NodeOrInt::Int(30)]
            ),
            Vec::from([NodeOrInt::Int(0), NodeOrInt::Int(0), NodeOrInt::Int(0)]),
            "Test multiple size-1 dimensions"
        );

        assert_eq!(
            filter_strides(
                &[
                    NodeOrInt::Int(2),
                    NodeOrInt::Int(1),
                    NodeOrInt::Int(3),
                    NodeOrInt::Int(1)
                ],
                &[
                    NodeOrInt::Int(12),
                    NodeOrInt::Int(4),
                    NodeOrInt::Int(1),
                    NodeOrInt::Int(10)
                ]
            ),
            Vec::from([
                NodeOrInt::Int(12),
                NodeOrInt::Int(0),
                NodeOrInt::Int(1),
                NodeOrInt::Int(0)
            ]),
            "Test mixed dimensions"
        );
    }

    #[test]
    fn test_strides_for_shape() {
        assert_eq!(strides_for_shape(&[]), Vec::new(), "Test empty shape");

        assert_eq!(
            strides_for_shape(&[NodeOrInt::Int(5)]),
            Vec::from([NodeOrInt::Int(1)]),
            "Test single dimension"
        );

        assert_eq!(
            strides_for_shape(&[NodeOrInt::Int(3), NodeOrInt::Int(4)]),
            Vec::from([NodeOrInt::Int(4), NodeOrInt::Int(1)]),
            "Test two dimensions"
        );

        assert_eq!(
            strides_for_shape(&[NodeOrInt::Int(2), NodeOrInt::Int(3), NodeOrInt::Int(4)]),
            Vec::from([NodeOrInt::Int(12), NodeOrInt::Int(4), NodeOrInt::Int(1)]),
            "Test three dimensions"
        );

        assert_eq!(
            strides_for_shape(&[
                NodeOrInt::Int(2),
                NodeOrInt::Int(3),
                NodeOrInt::Int(4),
                NodeOrInt::Int(5)
            ]),
            Vec::from([
                NodeOrInt::Int(60),
                NodeOrInt::Int(20),
                NodeOrInt::Int(5),
                NodeOrInt::Int(1)
            ]),
            "Test four dimensions"
        );

        assert_eq!(
            strides_for_shape(&[
                NodeOrInt::Int(1),
                NodeOrInt::Int(2),
                NodeOrInt::Int(1),
                NodeOrInt::Int(3)
            ]),
            Vec::from([
                NodeOrInt::Int(0),
                NodeOrInt::Int(3),
                NodeOrInt::Int(0),
                NodeOrInt::Int(1)
            ]),
            "Test with size 1 dimensions"
        );

        assert_eq!(
            strides_for_shape(&[NodeOrInt::Int(1), NodeOrInt::Int(1), NodeOrInt::Int(1)]),
            Vec::from([NodeOrInt::Int(0), NodeOrInt::Int(0), NodeOrInt::Int(0)]),
            "Test with all size 1 dimensions"
        );

        assert_eq!(
            strides_for_shape(&[
                NodeOrInt::Int(2),
                NodeOrInt::Int(1),
                NodeOrInt::Int(3),
                NodeOrInt::Int(1)
            ]),
            Vec::from([
                NodeOrInt::Int(3),
                NodeOrInt::Int(0),
                NodeOrInt::Int(1),
                NodeOrInt::Int(0)
            ]),
            "Test with mixed dimensions including size 1"
        );
    }

    #[test]
    fn test_view_create() {
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]))
            .build()
            .unwrap();
        assert_eq!(v.shape, Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]));
        assert_eq!(v.strides, Vec::from([NodeOrInt::Int(3), NodeOrInt::Int(1)]));
        assert_eq!(v.offset, NodeOrInt::Int(0));
        assert!(v.mask.is_none());
        assert!(v.contiguous);
    }

    #[test]
    fn test_view_with_custom_strides() {
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]))
            .with_strides(Vec::from([NodeOrInt::Int(6), NodeOrInt::Int(2)]))
            .build()
            .unwrap();
        assert_eq!(v.shape, Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]));
        assert_eq!(v.strides, Vec::from([NodeOrInt::Int(6), NodeOrInt::Int(2)]));
        assert_eq!(v.offset, NodeOrInt::Int(0));
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_offset() {
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]))
            .with_offset(NodeOrInt::Int(5))
            .build()
            .unwrap();
        assert_eq!(v.shape, Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]));
        assert_eq!(v.strides, Vec::from([NodeOrInt::Int(3), NodeOrInt::Int(1)]));
        assert_eq!(v.offset, NodeOrInt::Int(5));
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_mask() {
        let mask = Vec::from([
            (NodeOrInt::Int(0), NodeOrInt::Int(1)),
            (NodeOrInt::Int(1), NodeOrInt::Int(2)),
        ]);
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]))
            .with_mask(mask.clone())
            .build()
            .unwrap();
        assert_eq!(v.shape, Vec::from([NodeOrInt::Int(2), NodeOrInt::Int(3)]));
        assert_eq!(v.strides, Vec::from([NodeOrInt::Int(3), NodeOrInt::Int(1)]));
        assert_eq!(v.offset, NodeOrInt::Int(0));
        assert_eq!(v.mask, Some(mask));
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_size_1_dimensions() {
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(1), NodeOrInt::Int(3)]))
            .build()
            .unwrap();
        assert_eq!(
            v.shape,
            Vec::from([NodeOrInt::Int(1), NodeOrInt::Int(3)]),
            "shape"
        );
        assert_eq!(
            v.strides,
            Vec::from([NodeOrInt::Int(0), NodeOrInt::Int(1)]),
            "strides"
        );
        assert_eq!(v.offset, NodeOrInt::Int(0), "offset");
        assert!(v.mask.is_none(), "mask");
        assert!(v.contiguous, "contiguous");
    }

    #[test]
    fn test_view_with_custom_strides_and_size_1_dimensions() {
        let v = ViewBuilder::new()
            .with_shape(Vec::from([NodeOrInt::Int(1), NodeOrInt::Int(3)]))
            .with_strides(Vec::from([NodeOrInt::Int(10), NodeOrInt::Int(2)]))
            .build()
            .unwrap();
        assert_eq!(v.shape, Vec::from([NodeOrInt::Int(1), NodeOrInt::Int(3)]));
        assert_eq!(v.strides, Vec::from([NodeOrInt::Int(0), NodeOrInt::Int(2)]));
        assert_eq!(v.offset, NodeOrInt::Int(0));
        assert!(v.mask.is_none());
        assert!(!v.contiguous);
    }

    #[test]
    fn test_view_with_empty_shape() {
        let v = ViewBuilder::new().with_shape(Vec::new()).build().unwrap();
        assert_eq!(v.shape, Vec::new());
        assert_eq!(v.strides, Vec::new());
        assert_eq!(v.offset, NodeOrInt::Int(0));
        assert!(v.mask.is_none());
        assert!(v.contiguous);
    }
}
