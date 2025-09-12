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

use z3::ast::Int;

use crate::{
    error::Error,
    fxgraph::{shape::SymInt, types::Type},
};

pub fn create_symint_ty(symint: &SymInt) -> Result<Type, Error> {
    let symint_ty = match symint {
        SymInt::Int(value) => Int::from_i64(*value),
        SymInt::Sym(value) => Int::new_const(value.as_str()),
    };

    Ok(symint_ty.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_symint_ty() {
        let symint = SymInt::Int(1);
        let symint_ty = create_symint_ty(&symint).unwrap();

        assert_eq!(symint_ty, Int::from_i64(1));
    }

    #[test]
    fn test_create_symint_ty_sym() {
        let symint = SymInt::Sym("x".to_string());
        let symint_ty = create_symint_ty(&symint).unwrap();

        assert_eq!(symint_ty, Int::new_const("x"));
    }
}
