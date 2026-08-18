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

//! Thread-local name scope for annotating graph nodes with human-readable names.
//!
//! Model builders push scope segments via [`name_scope`], which returns a
//! [`NameGuard`] that pops the segment on drop.  [`Graph::add_node`] calls
//! [`current_scope`] to capture the full dotted path at recording time.
//!
//! Example (ultralytics-compatible naming):
//! ```
//! let _g = name_scope("model.0.conv");
//! // nodes recorded here get name "model.0.conv"
//! let _g2 = name_scope("weight"); // nested: "model.0.conv.weight"
//! ```

use alloc::{string::String, vec::Vec};

thread_local! {
    static SCOPE: core::cell::RefCell<Vec<String>> = core::cell::RefCell::new(Vec::new());
}

/// RAII guard that pops its scope segment when dropped.
pub struct NameGuard;

impl Drop for NameGuard {
    fn drop(&mut self) {
        SCOPE.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Push `name` onto the thread-local scope stack and return a guard that pops
/// it on drop.  Nest calls to build dotted paths.
pub fn name_scope(name: impl Into<String>) -> NameGuard {
    SCOPE.with(|s| s.borrow_mut().push(name.into()));
    NameGuard
}

/// Return the current dotted scope path, or `None` if the stack is empty.
pub fn current_scope() -> Option<String> {
    SCOPE.with(|s| {
        let s = s.borrow();
        if s.is_empty() {
            None
        } else {
            Some(s.join("."))
        }
    })
}
