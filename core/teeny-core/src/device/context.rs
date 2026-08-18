/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use alloc::vec::Vec;

use crate::{
    device::{Device, hardware::HardwareProfile},
    errors::Result,
};

/// A device's identifying metadata.
pub trait DeviceInfo: Sized {
    /// This device's ID type.
    type Id;

    /// This device's ID.
    fn id(&self) -> Self::Id;
    /// This device's name.
    fn name(&self) -> &str;
    /// This device's hardware profile (memory hierarchy, compute-unit
    /// count, ...), for tile-shape / scheduling cost models. See
    /// [`crate::device::hardware`].
    fn hardware_profile(&self) -> HardwareProfile;
}

/// The entry point for discovering and opening devices.
pub trait Context<'a> {
    /// The device type this context opens.
    type Device: Device<'a>;
    /// The device-info type returned by [`Context::list_devices`].
    type DeviceInfo: DeviceInfo;

    /// Lists all available devices.
    fn list_devices(&self) -> Result<Vec<Self::DeviceInfo>>;

    /// Opens the device with the given ID.
    fn device(&self, id: &<Self::DeviceInfo as DeviceInfo>::Id) -> Result<Self::Device>;
}
