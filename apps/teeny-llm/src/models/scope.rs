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

//! [`Scope`]: the dotted names a model's graph builder gives its nodes.

/// A dotted prefix for node names, passed down a model's graph builder so each node that
/// holds weights is named after its checkpoint tensors (`model.layers.3.self_attn.q_proj`,
/// to which a compiled model appends each parameter slot's name, `.weight`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Scope(String);

impl Scope {
    /// The scope named `name` at the top level.
    pub(crate) fn root(name: &str) -> Self {
        Self(name.to_owned())
    }

    /// The scope `segment` inside this one.
    pub(crate) fn child(&self, segment: impl AsRef<str>) -> Self {
        Self(self.name(segment.as_ref()))
    }

    /// The name of `leaf` inside this scope.
    pub(crate) fn name(&self, leaf: &str) -> String {
        if self.0.is_empty() {
            leaf.to_owned()
        } else {
            format!("{}.{leaf}", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scopes_join_with_dots() {
        let layer = Scope::root("model").child("layers.3");
        assert_eq!(
            layer.child("self_attn").name("q_proj"),
            "model.layers.3.self_attn.q_proj"
        );
        assert_eq!(Scope::default().name("lm_head"), "lm_head");
    }
}
