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

use super::{Node, NodeOps, NodeOrInt};

pub struct VarNode {
    expr: String,
    min: i64,
    max: i64,
}

impl NodeOps for VarNode {
    fn b(&self) -> NodeOrInt {
        NodeOrInt::Node(Node::Var(self))
    }

    fn min(&self) -> i64 {
        self.min
    }

    fn max(&self) -> i64 {
        self.max
    }
}

impl std::ops::Add for VarNode {
    type Output = Node;
    fn add(self, rhs: Self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("({} + {})", self.expr, rhs.expr),
            min: self.min + rhs.min,
            max: self.max + rhs.max,
        })
    }
}

impl std::ops::Sub for VarNode {
    type Output = Node;
    fn sub(self, rhs: Self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("({} - {})", self.expr, rhs.expr),
            min: self.min - rhs.max,
            max: self.max - rhs.min,
        })
    }
}

impl std::ops::Mul for VarNode {
    type Output = Node;
    fn mul(self, rhs: Self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("({} * {})", self.expr, rhs.expr),
            min: self.min * rhs.min,
            max: self.max * rhs.max,
        })
    }
}

impl std::ops::Div for VarNode {
    type Output = Node;
    fn div(self, rhs: Self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("({} / {})", self.expr, rhs.expr),
            min: self.min / rhs.max,
            max: self.max / rhs.min,
        })
    }
}

impl std::ops::Rem for VarNode {
    type Output = Node;
    fn rem(self, rhs: Self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("({} % {})", self.expr, rhs.expr),
            min: 0,
            max: rhs.max - 1,
        })
    }
}

impl std::ops::Neg for VarNode {
    type Output = Node;
    fn neg(self) -> Self::Output {
        Node::Var(VarNode {
            expr: format!("-{}", self.expr),
            min: -self.max,
            max: -self.min,
        })
    }
}
