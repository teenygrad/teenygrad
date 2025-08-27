/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
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

use teeny_core::fxgraph::FXGraph;

use crate::{error::Error, torch::Output};

pub fn output<'a>(fxgraph: &mut FXGraph, node: &Output<'a>) -> Result<(), Error> {
    todo!("output: {:?}", node);
    // let args = node
    //     .args()
    //     .ok_or_else(|| Error::NoGraphNodeArgs(format!("{node:?}")))?
    //     .iter()
    //     .collect::<Vec<_>>();
    // if args.len() != 1 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // if args[0].starts_with("(") && args[0].ends_with(")") {
    //     let args = args[0][1..args[0].len() - 1]
    //         .split(",")
    //         .map(|x| find_or_create(fxgraph, x))
    //         .collect::<Vec<_>>();
    //     let name = node
    //         .name()
    //         .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;

    //     fxgraph.outputs.extend(args.clone());
    //     fxgraph.add_operation(name, FxGraphLang::Output(args));
    // } else {
    //     return Err(Error::GraphNodeInvalidArgs(format!("output - {:?}", node)));
    // }

    // Ok(())
}
