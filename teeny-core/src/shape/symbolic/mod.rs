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
use hashbrown::HashMap;
use var_node::VarNode;

mod var_node;
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

    //   def expand(self, idxs:Optional[Tuple[VariableOrNum, ...]]=None) -> List[Node]:
    //     if idxs is None: idxs = (self.expand_idx(),)
    //     return [self.substitute(dict(zip(idxs, (NumNode(x) for x in rep)))) for rep in Node.iter_idxs(idxs)]
    fn expand(&self, idxs: Option<Vec<VarOrNum>>) -> Vec<Node> {
        let _idxs = idxs.or_else(|| Some(Vec::from([self.expand_idx()])));
        todo!()
    }

    fn substitute(&self, _var_vals: HashMap<VarOrNum, Node>) -> Node {
        todo!()
    }

    fn unbind(&self) -> (Node, Option<i64>) {
        todo!()
    }

    fn key(&self) -> String {
        todo!()
    }

    fn hash(&self) -> u64 {
        todo!()
    }

    fn bool(&self) -> bool {
        todo!()
    }

    fn to_string(&self) -> String {
        self.key()
    }

    fn mul(&self, _other: Node) -> Node {
        todo!()
    }
}

//   def iter_idxs(idxs:Tuple[VariableOrNum, ...]) -> Iterator[Tuple[int,...]]:
//     yield from (x[::-1] for x in product(*[[x for x in range(v.min, v.max + 1)] for v in idxs[::-1]]))

pub fn iter_idxs(_idxs: Vec<VarOrNum>) {
    todo!()
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Node {
    Var(VarNode),
    Num(i64),
}

pub enum NodeOrInt {
    Node(Node),
    Int(i64),
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum VarOrNum {
    Var(VarNode),
    Num(i64),
}
