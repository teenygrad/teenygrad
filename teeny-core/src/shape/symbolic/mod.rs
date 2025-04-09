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

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use var_node::VarNode;

mod var_node;
pub trait NodeOps:
    Sized
    + Add<Output = Node>
    + Sub<Output = Node>
    + Mul<Output = Node>
    + Div<Output = Node>
    + Rem<Output = Node>
    + Neg<Output = Node>
{
    fn b() -> NodeOrInt;
    fn min() -> i64;
    fn max() -> i64;
}

pub enum Node {
    Var(VarNode),
}

pub enum NodeOrInt {
    Node(Node),
    Int(i64),
}
