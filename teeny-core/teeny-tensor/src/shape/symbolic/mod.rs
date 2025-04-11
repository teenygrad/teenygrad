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

use alloc::vec::Vec;
use var_node::VarNode;

pub mod var_node;
pub trait NodeOps: Sized {
    fn b(&self) -> NodeOrInt;
    fn min(&self) -> i64;
    fn max(&self) -> i64;

    fn vars(&self) -> Vec<VarNode> {
        Vec::new()
    }

    //   def expand_idx(self) -> VariableOrNum: return next((v for v in self.vars() if v.expr is None), NumNode(0))
    fn expand_idx(&self) -> VarOrNum {
        self.vars()
            .into_iter()
            .find(|v| v.expr.is_none())
            .map(VarOrNum::Var)
            .unwrap_or(VarOrNum::Num(0))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Var(VarNode),
    Num(i64),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeOrInt {
    Node(Node),
    Int(i64),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VarOrNum {
    Var(VarNode),
    Num(i64),
}
