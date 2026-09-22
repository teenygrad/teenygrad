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

//! Just enough ELF64 reading to find a kernel library's entry point without loading it: the
//! library is a RISC-V ELF, which a process on another architecture can't `dlopen`.

use std::path::Path;

use anyhow::Context;

use crate::errors::{Error, Result};

const SHT_DYNSYM: u32 = 11;
const STT_FUNC: u8 = 2;
const SHN_UNDEF: u16 = 0;

/// The suffix `teeny-macros` gives every kernel's C-ABI entry point.
const ENTRY_POINT_SUFFIX: &str = "_entry_point";

/// The name of the single function the ELF64 shared library at `path` exports whose name ends in
/// `_entry_point`.
pub(crate) fn find_entry_point(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read kernel library {}", path.display()))?;
    let functions = exported_functions(&bytes).ok_or_else(|| Error::InvalidKernelLibrary {
        path: path.to_path_buf(),
    })?;

    let mut entry_points = functions
        .into_iter()
        .filter(|name| name.ends_with(ENTRY_POINT_SUFFIX));
    match (entry_points.next(), entry_points.next()) {
        (Some(name), None) => Ok(name),
        _ => Err(Error::EntryPointNotFound {
            path: path.to_path_buf(),
        }
        .into()),
    }
}

/// The names of the functions defined in a little-endian ELF64 file's dynamic symbol table, or
/// `None` if `bytes` isn't such a file.
fn exported_functions(bytes: &[u8]) -> Option<Vec<String>> {
    const ELFCLASS64: u8 = 2;
    const ELFDATA2LSB: u8 = 1;
    if bytes.get(..4)? != b"\x7fELF"
        || *bytes.get(4)? != ELFCLASS64
        || *bytes.get(5)? != ELFDATA2LSB
    {
        return None;
    }

    let section_offset = usize::try_from(read_u64(bytes, 0x28)?).ok()?;
    let section_size = usize::from(read_u16(bytes, 0x3a)?);
    let section_count = usize::from(read_u16(bytes, 0x3c)?);
    let section = |index: usize| {
        SectionHeader::parse(
            bytes,
            section_offset.checked_add(index.checked_mul(section_size)?)?,
        )
    };

    let symbols = (0..section_count)
        .filter_map(section)
        .find(|header| header.kind == SHT_DYNSYM)?;
    let strings = section(usize::try_from(symbols.link).ok()?)?;
    let strings = bytes.get(strings.offset..strings.offset.checked_add(strings.size)?)?;
    if symbols.entry_size == 0 {
        return None;
    }

    let mut names = Vec::new();
    for index in 0..symbols.size / symbols.entry_size {
        let symbol = symbols
            .offset
            .checked_add(index.checked_mul(symbols.entry_size)?)?;
        let name_offset = usize::try_from(read_u32(bytes, symbol)?).ok()?;
        let kind = *bytes.get(symbol + 4)? & 0xf;
        let section_index = read_u16(bytes, symbol + 6)?;
        if kind == STT_FUNC && section_index != SHN_UNDEF {
            let name = strings.get(name_offset..)?;
            let end = name.iter().position(|&b| b == 0)?;
            names.push(String::from_utf8_lossy(&name[..end]).into_owned());
        }
    }
    Some(names)
}

/// The fields of an ELF64 section header this module needs.
struct SectionHeader {
    kind: u32,
    offset: usize,
    size: usize,
    link: u32,
    entry_size: usize,
}

impl SectionHeader {
    fn parse(bytes: &[u8], at: usize) -> Option<Self> {
        Some(Self {
            kind: read_u32(bytes, at.checked_add(0x04)?)?,
            offset: usize::try_from(read_u64(bytes, at.checked_add(0x18)?)?).ok()?,
            size: usize::try_from(read_u64(bytes, at.checked_add(0x20)?)?).ok()?,
            link: read_u32(bytes, at.checked_add(0x28)?)?,
            entry_size: usize::try_from(read_u64(bytes, at.checked_add(0x38)?)?).ok()?,
        })
    }
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejects_bytes_that_are_not_elf64() {
        assert!(exported_functions(b"not an ELF file at all").is_none());
    }

    #[test]
    fn test_reports_a_library_without_an_entry_point() {
        // The test binary itself is a valid ELF64 file that exports no kernel entry point.
        let err = find_entry_point(Path::new("/proc/self/exe")).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<Error>(),
            Some(Error::EntryPointNotFound { .. })
        ));
    }
}
