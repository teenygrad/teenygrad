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
