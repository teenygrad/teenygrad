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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::driver::Driver;
use crate::error::DriverError;
use crate::error::Result;

#[allow(clippy::type_complexity)]
static DRIVERS: OnceLock<RwLock<HashMap<String, Arc<Mutex<dyn Driver + Send + Sync>>>>> =
    OnceLock::new();

pub struct DriverManager;

impl DriverManager {
    pub fn register(driver: Arc<Mutex<dyn Driver + Send + Sync>>) {
        let drivers = DRIVERS.get_or_init(|| RwLock::new(HashMap::new()));
        let id = driver.lock().unwrap().id().to_string();
        drivers.write().unwrap().insert(id, driver);
    }

    pub fn driver(id: &str) -> Result<Option<Arc<Mutex<dyn Driver + Send + Sync>>>> {
        if let Some(drivers) = DRIVERS.get() {
            let drivers = drivers
                .read()
                .map_err(|e| DriverError::LockError(e.to_string()))?;

            Ok(drivers.get(id).cloned())
        } else {
            Err(DriverError::NotFound(id.to_string()))
        }
    }

    pub fn drivers() -> Result<Vec<Arc<Mutex<dyn Driver + Send + Sync>>>> {
        let drivers = DRIVERS.get().map(|d| d.read().unwrap());
        if let Some(drivers) = drivers {
            Ok(drivers.values().cloned().collect())
        } else {
            Err(DriverError::NotFound("No drivers registered".to_string()))
        }
    }
}
