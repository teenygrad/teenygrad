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

//! Assigns a kernel's arguments to RISC-V `lp64d` calling-convention slots.
//!
//! A kernel's argument list is only known at run time (from [`KernelArgs::visit_args`]), so it
//! can't be called through a Rust function type written for it. Instead [`Frame::new`] lays the
//! arguments out exactly as the `lp64d` ABI would for a call to the kernel's real signature --
//! integer registers, floating-point registers, then the stack -- and the native launcher (built
//! only on `riscv64`) calls the entry point through one fixed signature that fills all eight
//! `a` registers, all eight `fa` registers and [`STACK_SLOTS`] stack slots from a [`Frame`]. The
//! callee reads only the slots its real signature uses; the rest are ignored, which the ABI
//! allows because the caller owns (and pops) the outgoing argument area.
//!
//! The layout itself is plain data, so it is unit-tested on every host; only the call needs
//! RISC-V.

use teeny_core::device::program::ArgVisitor;

/// Number of integer (`a0`-`a7`) and floating-point (`fa0`-`fa7`) argument registers.
pub(crate) const ARG_REGS: usize = 8;

/// Stack slots the fixed call signature passes. Generous for a Triton kernel: pointers and
/// scalars past the first eight integer arguments, plus the six program-id/grid arguments
/// TritonCPU appends.
pub(crate) const STACK_SLOTS: usize = 32;

/// One kernel argument, in the `lp64d` classes that decide where it goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Slot {
    /// An integer-class value (pointer, bool, integer), already widened to 64 bits.
    Int(u64),
    /// An `f32`.
    F32(f32),
    /// An `f64`.
    F64(f64),
}

/// A kernel call's arguments, assigned to registers and stack slots.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Frame {
    /// `a0`-`a7`.
    pub(crate) int_regs: [u64; ARG_REGS],
    /// `fa0`-`fa7`, as raw 64-bit register contents (an `f32` is NaN-boxed).
    pub(crate) float_regs: [u64; ARG_REGS],
    /// The outgoing stack argument area, one 8-byte slot per spilled argument.
    pub(crate) stack: [u64; STACK_SLOTS],
}

impl Frame {
    /// Lays out `args` followed by TritonCPU's program-id/grid arguments (`pid_x`, `pid_y`,
    /// `pid_z` as `i32`, then `grid_x`, `grid_y`, `grid_z` as `u32`), with the program-id slots
    /// zeroed for [`Self::set_program_id`] to fill in.
    ///
    /// Returns `None` if the arguments need more than [`STACK_SLOTS`] stack slots.
    pub(crate) fn new(args: &[Slot], grid: [u32; 3]) -> Option<(Frame, ProgramIdSlots)> {
        let mut frame = Frame {
            int_regs: [0; ARG_REGS],
            float_regs: [0; ARG_REGS],
            stack: [0; STACK_SLOTS],
        };
        let mut next_int = 0;
        let mut next_float = 0;
        let mut next_stack = 0;

        let mut place = |slot: Slot| -> Option<Location> {
            // A float goes in the next `fa` register; once those run out it is passed like an
            // integer of the same width (psABI "Hardware Floating-point Calling Convention").
            let bits = match slot {
                Slot::F32(v) if next_float < ARG_REGS => {
                    frame.float_regs[next_float] = nan_box(v);
                    next_float += 1;
                    return Some(Location::FloatReg(next_float - 1));
                }
                Slot::F64(v) if next_float < ARG_REGS => {
                    frame.float_regs[next_float] = v.to_bits();
                    next_float += 1;
                    return Some(Location::FloatReg(next_float - 1));
                }
                Slot::F32(v) => u64::from(v.to_bits()),
                Slot::F64(v) => v.to_bits(),
                Slot::Int(v) => v,
            };
            if next_int < ARG_REGS {
                frame.int_regs[next_int] = bits;
                next_int += 1;
                Some(Location::IntReg(next_int - 1))
            } else if next_stack < STACK_SLOTS {
                frame.stack[next_stack] = bits;
                next_stack += 1;
                Some(Location::Stack(next_stack - 1))
            } else {
                None
            }
        };

        for &arg in args {
            place(arg)?;
        }
        let pid = [
            place(Slot::Int(0))?,
            place(Slot::Int(0))?,
            place(Slot::Int(0))?,
        ];
        for n in grid {
            place(int_u32(n))?;
        }
        Some((frame, ProgramIdSlots(pid)))
    }

    /// Sets the program id this frame calls the kernel with.
    pub(crate) fn set_program_id(&mut self, slots: &ProgramIdSlots, pid: [u32; 3]) {
        for (location, id) in slots.0.iter().zip(pid) {
            let value = int_i32(id as i32);
            match *location {
                Location::IntReg(i) => self.int_regs[i] = value,
                Location::Stack(i) => self.stack[i] = value,
                Location::FloatReg(_) => unreachable!("program ids are integer-class"),
            }
        }
    }
}

/// Where [`Frame::new`] placed the three program-id arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ProgramIdSlots([Location; 3]);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Location {
    IntReg(usize),
    FloatReg(usize),
    Stack(usize),
}

/// An `f32` in a 64-bit FP register: the upper 32 bits must be all ones, or a single-precision
/// instruction reading the register sees the canonical NaN instead of the value.
fn nan_box(v: f32) -> u64 {
    0xffff_ffff_0000_0000 | u64::from(v.to_bits())
}

/// The psABI widens every integer narrower than 32 bits to 32 bits according to its own
/// signedness, then sign-extends the 32-bit value to 64 bits -- so a `u32` is sign-extended too.
fn int_i32(v: i32) -> u64 {
    i64::from(v) as u64
}

fn int_u32(v: u32) -> Slot {
    Slot::Int(int_i32(v as i32))
}

/// Collects a kernel's arguments as [`Slot`]s.
#[derive(Default)]
pub(crate) struct SlotCollector(pub(crate) Vec<Slot>);

impl SlotCollector {
    /// The `lp64d` slots for `args`.
    #[cfg(target_arch = "riscv64")]
    pub(crate) fn collect<A: teeny_core::device::program::KernelArgs>(args: &A) -> Vec<Slot> {
        let mut collector = Self::default();
        args.visit_args(&mut collector);
        collector.0
    }
}

impl ArgVisitor for SlotCollector {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void) {
        self.0.push(Slot::Int(ptr as u64));
    }
    fn visit_bool(&mut self, val: bool) {
        self.0.push(Slot::Int(u64::from(val)));
    }
    fn visit_i8(&mut self, val: i8) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_i16(&mut self, val: i16) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_i32(&mut self, val: i32) {
        self.0.push(Slot::Int(int_i32(val)));
    }
    fn visit_i64(&mut self, val: i64) {
        self.0.push(Slot::Int(val as u64));
    }
    fn visit_u8(&mut self, val: u8) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_u16(&mut self, val: u16) {
        self.0.push(Slot::Int(int_i32(val.into())));
    }
    fn visit_u32(&mut self, val: u32) {
        self.0.push(int_u32(val));
    }
    fn visit_u64(&mut self, val: u64) {
        self.0.push(Slot::Int(val));
    }
    fn visit_f32(&mut self, val: f32) {
        self.0.push(Slot::F32(val));
    }
    fn visit_f64(&mut self, val: f64) {
        self.0.push(Slot::F64(val));
    }
}

#[cfg(target_arch = "riscv64")]
pub(crate) use launcher::launch;

#[cfg(target_arch = "riscv64")]
mod launcher {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use anyhow::{Result, anyhow, bail};
    use teeny_core::device::program::KernelArgs;

    use super::{Frame, STACK_SLOTS, SlotCollector};
    use crate::runtime::KernelLibrary;

    /// Every kernel entry point, called with all eight `a` registers, all eight `fa` registers
    /// and [`STACK_SLOTS`] stack slots filled from a [`Frame`].
    #[rustfmt::skip]
    type Entry = unsafe extern "C" fn(
        u64, u64, u64, u64, u64, u64, u64, u64,
        f64, f64, f64, f64, f64, f64, f64, f64,
        u64, u64, u64, u64, u64, u64, u64, u64,
        u64, u64, u64, u64, u64, u64, u64, u64,
        u64, u64, u64, u64, u64, u64, u64, u64,
        u64, u64, u64, u64, u64, u64, u64, u64,
    );
    const _: () = assert!(STACK_SLOTS == 32);

    /// Calls `frame`'s entry point. The FP registers are passed as `f64`s carrying the frame's
    /// raw bits: moving a value between registers never changes its bits, NaN-boxed `f32`s
    /// included.
    ///
    /// # Safety
    /// `entry` must be a kernel entry point whose arguments `frame` was laid out for.
    #[rustfmt::skip]
    unsafe fn call(entry: Entry, frame: &Frame) {
        let a = &frame.int_regs;
        let f = frame.float_regs.map(f64::from_bits);
        let s = &frame.stack;
        unsafe {
            entry(
                a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15],
                s[16], s[17], s[18], s[19], s[20], s[21], s[22], s[23],
                s[24], s[25], s[26], s[27], s[28], s[29], s[30], s[31],
            )
        }
    }

    /// Kernel libraries this process has loaded, by path. A library stays loaded for the rest
    /// of the process, so every launch after the first is just a lookup.
    #[derive(Default)]
    pub(crate) struct Libraries(Mutex<HashMap<PathBuf, Arc<KernelLibrary>>>);

    impl Libraries {
        fn get(&self, path: &Path) -> Result<Arc<KernelLibrary>> {
            let mut libraries = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(library) = libraries.get(path) {
                return Ok(Arc::clone(library));
            }
            load_libm()?;
            let library = Arc::new(KernelLibrary::load(path)?);
            libraries.insert(path.to_path_buf(), Arc::clone(&library));
            Ok(library)
        }
    }

    /// Kernels call libm (e.g. `expf`) without listing it as a dependency, so make its symbols
    /// globally available before loading one -- as the QEMU trampoline does.
    fn load_libm() -> Result<()> {
        static LIBM: OnceLock<std::result::Result<(), String>> = OnceLock::new();
        LIBM.get_or_init(|| {
            use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
            // Safety: libm's initializers are the system's own.
            let libm = unsafe { Library::open(Some("libm.so.6"), RTLD_NOW | RTLD_GLOBAL) }
                .map_err(|e| format!("dlopen(libm.so.6) failed: {e}"))?;
            std::mem::forget(libm);
            Ok(())
        })
        .clone()
        .map_err(|e| anyhow!(e))
    }

    /// The number of threads a launch spreads its program ids over: `TEENY_RISCV_THREADS`, or
    /// every available core.
    fn threads() -> usize {
        static THREADS: OnceLock<usize> = OnceLock::new();
        *THREADS.get_or_init(|| {
            std::env::var("TEENY_RISCV_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .or_else(|| std::thread::available_parallelism().ok().map(Into::into))
                .unwrap_or(1)
                .max(1)
        })
    }

    /// Runs the kernel `entry_point` from `library` once for every program id in `grid`, with
    /// `args`, in this process.
    ///
    /// Program ids are independent (as on a GPU), so they run in parallel on [`threads`]
    /// threads. Every pointer in `args` addresses host memory the kernel reads and writes
    /// directly.
    pub(crate) fn launch<A: KernelArgs>(
        libraries: &Libraries,
        library: &Path,
        entry_point: &str,
        grid: [u32; 3],
        args: &A,
    ) -> Result<()> {
        let slots = SlotCollector::collect(args);
        let Some((frame, pid_slots)) = Frame::new(&slots, grid) else {
            bail!(
                "{entry_point} takes {} arguments, more than the native RISC-V launcher passes \
                 on the stack ({STACK_SLOTS} slots)",
                slots.len()
            );
        };

        let library = libraries.get(library)?;
        // Safety: the entry point TritonCPU generated takes the kernel's own arguments, then the
        // program id and grid, which is exactly what `frame` was laid out for.
        let entry: Entry = unsafe { std::mem::transmute(library.symbol(entry_point)?) };

        let [gx, gy, gz] = grid.map(u64::from);
        let total = gx * gy * gz;
        let threads = (threads() as u64).min(total).max(1);
        let next = AtomicU64::new(0);
        let run = || {
            let mut frame = frame.clone();
            // Hand out program ids in small chunks so uneven per-id costs still balance.
            let chunk = (total / (threads * 8)).max(1);
            loop {
                let start = next.fetch_add(chunk, Ordering::Relaxed);
                if start >= total {
                    break;
                }
                for id in start..(start + chunk).min(total) {
                    let pid = [id % gx, (id / gx) % gy, id / (gx * gy)].map(|n| n as u32);
                    frame.set_program_id(&pid_slots, pid);
                    // Safety: see `entry` above.
                    unsafe { call(entry, &frame) };
                }
            }
        };
        if threads == 1 {
            run();
        } else {
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(run);
                }
            });
        }
        Ok(())
    }
}

#[cfg(target_arch = "riscv64")]
pub(crate) use launcher::Libraries;

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(args: &[Slot]) -> Frame {
        let (mut frame, pid) = Frame::new(args, [4, 5, 6]).unwrap();
        frame.set_program_id(&pid, [1, 2, 3]);
        frame
    }

    #[test]
    fn integers_fill_a_registers_then_the_stack() {
        let args: Vec<Slot> = (10..14).map(Slot::Int).collect();
        let frame = layout(&args);
        // 4 arguments, then pid x/y/z and grid x/y/z.
        assert_eq!(frame.int_regs, [10, 11, 12, 13, 1, 2, 3, 4]);
        assert_eq!(&frame.stack[..3], &[5, 6, 0]);
    }

    #[test]
    fn floats_use_fa_registers_independently_of_integers() {
        let frame = layout(&[Slot::Int(7), Slot::F32(1.5), Slot::F64(2.5), Slot::Int(8)]);
        assert_eq!(&frame.int_regs[..2], &[7, 8]);
        assert_eq!(
            frame.float_regs[0],
            0xffff_ffff_0000_0000 | u64::from(1.5f32.to_bits())
        );
        assert_eq!(frame.float_regs[1], 2.5f64.to_bits());
        assert_eq!(frame.int_regs[2..5], [1, 2, 3]);
    }

    #[test]
    fn floats_past_fa7_go_to_a_registers_then_the_stack() {
        let mut args: Vec<Slot> = (0..ARG_REGS).map(|i| Slot::F32(i as f32)).collect();
        args.push(Slot::F32(9.0));
        args.extend((0..7).map(Slot::Int));
        args.push(Slot::F64(10.0));
        let frame = layout(&args);
        assert_eq!(frame.int_regs[0], u64::from(9.0f32.to_bits()));
        assert_eq!(frame.int_regs[1..], [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(frame.stack[0], 10.0f64.to_bits());
        // Then the program id and grid.
        assert_eq!(&frame.stack[1..7], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn narrow_integers_are_sign_extended_from_32_bits() {
        let mut collector = SlotCollector::default();
        collector.visit_i8(-1);
        collector.visit_u8(0xff);
        collector.visit_i32(-2);
        collector.visit_u32(0x8000_0000);
        collector.visit_bool(true);
        assert_eq!(
            collector.0,
            [
                Slot::Int(u64::MAX),
                Slot::Int(0xff),
                Slot::Int(-2i64 as u64),
                Slot::Int(0xffff_ffff_8000_0000),
                Slot::Int(1),
            ]
        );
    }

    #[test]
    fn too_many_arguments_is_an_error() {
        let args: Vec<Slot> = (0..(ARG_REGS + STACK_SLOTS) as u64)
            .map(Slot::Int)
            .collect();
        assert!(Frame::new(&args, [1, 1, 1]).is_none());
    }
}
