/*
 * Copyright (c) 2026 Teenygrad.
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

/// RISC-V chip identifier passed to `teenyc` as `-C target-cpu` when compiling for the
/// `riscv64-generic` target (see `rustc_target::spec::targets::riscv64_generic` and
/// `rustc_codegen_llvm::mlir::target::resolve` in the `teeny` compiler fork).
///
/// Unlike [`teeny_cuda::compiler::target::Capability`]'s `sm_NN` values, these are **not** LLVM
/// `-mcpu` names: the RISC-V backend (`RiscvBackend`) doesn't yet map them to real LLVM
/// cpu/feature strings, so codegen currently always targets a fixed generic RV64GC baseline
/// regardless of which variant is requested here. Once that mapping exists, this is the type
/// that should carry it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Generic RVV 1.0-capable core: no specific chip, just "has the V extension".
    GenericRvv1_0,
    /// SpacemiT K1 (e.g. Milk-V Duo, early SpacemiT boards).
    SpacemitK1,
    /// SpacemiT K3 (e.g. Milk-V Jupiter, Banana Pi BPI-F3).
    SpacemitK3,
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::GenericRvv1_0 => "generic-rvv1.0",
            Self::SpacemitK1 => "spacemit-k1",
            Self::SpacemitK3 => "spacemit-k3",
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for Capability {
    type Err = String;

    /// Accepts the canonical hyphenated form, case-insensitively.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "generic-rvv1.0" | "generic-rvv1_0" => Ok(Self::GenericRvv1_0),
            "spacemit-k1" => Ok(Self::SpacemitK1),
            "spacemit-k3" => Ok(Self::SpacemitK3),
            other => Err(format!(
                "unknown RISC-V capability '{other}'; expected one of generic-rvv1.0, spacemit-k1, spacemit-k3"
            )),
        }
    }
}

/// A RISC-V compilation target: a single chip [`Capability`].
///
/// Mirrors `teeny_cuda::compiler::target::Target`'s shape so call sites that are generic over
/// [`teeny_core::compiler::Target`] work the same way for either backend.
pub struct Target {
    /// The target chip's capability.
    pub capability: Capability,
}

impl Target {
    /// Creates a target for the given chip `capability`.
    pub fn new(capability: Capability) -> Self {
        Self { capability }
    }
}

impl teeny_core::compiler::Target for Target {
    fn target_cpu(&self) -> Option<String> {
        Some(self.capability.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Capability;

    #[test]
    fn accepts_canonical_forms() {
        assert_eq!(
            "generic-rvv1.0".parse::<Capability>().unwrap(),
            Capability::GenericRvv1_0
        );
        assert_eq!(
            "spacemit-k1".parse::<Capability>().unwrap(),
            Capability::SpacemitK1
        );
        assert_eq!(
            "SPACEMIT-K3".parse::<Capability>().unwrap(),
            Capability::SpacemitK3
        );
    }

    #[test]
    fn rejects_unknown_capability() {
        assert!("spacemit-k9".parse::<Capability>().is_err());
    }

    #[test]
    fn display_round_trips_through_from_str() {
        for cap in [
            Capability::GenericRvv1_0,
            Capability::SpacemitK1,
            Capability::SpacemitK3,
        ] {
            assert_eq!(cap.to_string().parse::<Capability>().unwrap(), cap);
        }
    }
}
