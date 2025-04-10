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
