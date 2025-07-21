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

use std::{hash::Hash, hash::Hasher};

use ksuid::Ksuid;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniqueId(pub Ksuid);

impl UniqueId {
    pub fn generate() -> Self {
        Self(Ksuid::generate())
    }
}

impl std::fmt::Debug for UniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_base62())
    }
}

impl Hash for UniqueId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_base62().hash(state);
    }
}
