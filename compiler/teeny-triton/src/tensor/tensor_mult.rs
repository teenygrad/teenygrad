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

use std::ops::{Mul, MulAssign};

use crate::{
    tensor::{DenseTensor, DynamicShape},
    types::NumericType,
};

impl<T: NumericType> Mul<T> for DenseTensor<DynamicShape, T> {
    type Output = DenseTensor<DynamicShape, T>;

    fn mul(self, _rhs: T) -> Self::Output {
        unimplemented!()
    }
}

impl<T: NumericType> Mul<DenseTensor<DynamicShape, T>> for DenseTensor<DynamicShape, T> {
    type Output = DenseTensor<DynamicShape, T>;

    fn mul(self, _rhs: DenseTensor<DynamicShape, T>) -> Self::Output {
        unimplemented!()
    }
}

impl<T: NumericType> MulAssign<DenseTensor<DynamicShape, T>> for DenseTensor<DynamicShape, T> {
    fn mul_assign(&mut self, _rhs: DenseTensor<DynamicShape, T>) {
        unimplemented!()
    }
}
