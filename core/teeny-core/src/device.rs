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

use once_cell::sync::OnceCell;
use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use crate::error::{Result, TeenyError};

pub trait Device: Send + Sync + std::fmt::Debug + Any {}

static CURRENT_DEVICE: OnceCell<Mutex<Option<Arc<dyn Device>>>> = OnceCell::new();

pub fn get_current_device() -> Result<std::sync::MutexGuard<'static, Arc<dyn Device>>> {
    CURRENT_DEVICE
        .get_or_init(|| Mutex::new(None))
        .try_lock()
        .map_err(|_| TeenyError::TryLockError("Failed to lock device".to_string()))
}

pub fn set_current_device(device: Arc<dyn Device>) -> Result<()> {
    let mut guard = CURRENT_DEVICE
        .get_or_init(|| Mutex::new(Some(device.clone())))
        .try_lock()
        .map_err(|_| TeenyError::TryLockError("Failed to lock device".to_string()))?;
    *guard = Some(device.clone());
    Ok(())
}
