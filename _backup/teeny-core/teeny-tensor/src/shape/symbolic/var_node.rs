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

use alloc::string::String;
use alloc::vec::Vec;

use super::{Node, NodeOps, NodeOrInt};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarNode {
    pub expr: Option<String>,
    pub val: Option<i64>,
    pub min: i64,
    pub max: i64,
}

impl VarNode {
    pub fn create(expr: Option<String>, min: i64, max: i64) -> Node {
        assert!(min >= 0 && min <= max);
        if min == max {
            return Node::Num(min);
        }

        Node::Var(Self {
            expr,
            min,
            max,
            val: None,
        })
    }

    pub fn bind(&mut self, val: i64) -> &mut Self {
        assert!(self.val.is_none());
        assert!(self.min <= val && val <= self.max);

        self.val = Some(val);
        self
    }

    pub fn unbind(&mut self) -> (VarNode, i64) {
        assert!(self.val.is_some());

        let val = self.val.unwrap();
        self.val = None;
        (self.clone(), val)
    }
}

impl NodeOps for VarNode {
    fn b(&self) -> NodeOrInt {
        todo!()
    }

    fn min(&self) -> i64 {
        self.min
    }

    fn max(&self) -> i64 {
        self.max
    }

    fn vars(&self) -> Vec<VarNode> {
        Vec::from([self.clone()])
    }
}
