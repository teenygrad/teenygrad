/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
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

use std::env;

use lrlex::lrlex_mod;
use lrpar::lrpar_mod;

use crate::toy_y::Expr;

// Using `lrlex_mod!` brings the lexer for `calc.l` into scope. By default the
// module name will be `calc_l` (i.e. the file name, minus any extensions,
// with a suffix of `_l`).
lrlex_mod!("toy_l");

// Using `lrpar_mod!` brings the parser for `calc.y` into scope. By default the
// module name will be `calc_y` (i.e. the file name, minus any extensions,
// with a suffix of `_y`).
lrpar_mod!("toy_y");

#[test]
fn test_toy() {
    // Get the `LexerDef` for the `calc` language.
    let lexerdef = toy_l::lexerdef();

    // Now we create a lexer with the `lexer` method with which
    // we can lex an input.
    let lexer = lexerdef.lexer("1 + 2");

    // Pass the lexer to the parser and lex and parse the input.
    let (res, errs) = toy_y::parse(&lexer);
    for e in errs {
        println!("{}", e.pp(&lexer, &toy_y::token_epp));
    }

    assert_eq!(
        format!("{:?}", res),
        "Some(Ok(Add { span: Span { start: 0, end: 5 }, lhs: Number { span: Span { start: 0, end: 1 } }, rhs: Number { span: Span { start: 4, end: 5 } } }))"
    );
}
