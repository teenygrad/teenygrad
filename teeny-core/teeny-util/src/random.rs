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

use std::sync::Mutex;

use lazy_static::lazy_static;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::Error;

lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::from_rng(&mut rand::rng()));
}

/// Sets the seed for the global random number generator
pub fn set_seed<'a>(seed: u64) -> Result<(), Error<'a>> {
    let mut rng = RNG.lock()?;
    *rng = StdRng::seed_from_u64(seed);
    Ok(())
}

pub fn randint<'a>(low: i64, high: i64, size: usize) -> Result<Vec<i64>, Error<'a>> {
    let mut rng = RNG.lock()?;
    let result = (0..size)
        .map(|_| rng.random_range(low..high))
        .collect::<Vec<_>>();

    Ok(result)
}
