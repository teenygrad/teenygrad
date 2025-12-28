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

#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![feature(no_core)]
#![feature(intrinsics, lang_items)]
#![feature(arbitrary_self_types)]
#![feature(const_trait_impl)]
#![feature(auto_traits)]
#![no_core]
#![no_implicit_prelude]

#[lang = "freeze"]
pub unsafe auto trait Freeze {}

#[lang = "meta_sized"]
pub trait MetaSized {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

// Required language items for no_core
#[lang = "sized"]
pub trait Sized: MetaSized + PointeeSized {}

#[lang = "clone"]
pub trait Clone {
    fn clone(&self) -> Self;
}

#[lang = "copy"]
pub trait Copy: Clone {}

impl<T> Copy for *const T {}
impl<T> Copy for *mut T {}
impl<T> Clone for *const T {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Clone for *mut T {
    fn clone(&self) -> Self {
        *self
    }
}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // This function is a shim that the compiler fills in
    unsafe { drop_in_place(to_drop) }
}

// Required language items for arithmetic operations
#[lang = "panic_const_add_overflow"]
pub fn panic_const_add_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_sub_overflow"]
pub fn panic_const_sub_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_mul_overflow"]
pub fn panic_const_mul_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_div_overflow"]
pub fn panic_const_div_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_rem_overflow"]
pub fn panic_const_rem_overflow() -> ! {
    loop {}
}

#[lang = "panic_location"]
pub struct PanicLocation {
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

// Also implement Copy for other primitive types that might be needed
impl Copy for i32 {}
impl Clone for i32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for f32 {}
impl Clone for f32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i8 {}
impl Clone for i8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i16 {}
impl Clone for i16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for i64 {}
impl Clone for i64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u8 {}
impl Clone for u8 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u16 {}
impl Clone for u16 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u32 {}
impl Clone for u32 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for u64 {}
impl Clone for u64 {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for bool {}
impl Clone for bool {
    fn clone(&self) -> Self {
        *self
    }
}

pub enum Option<T> {
    None,
    Some(T),
}

use Option::*;

pub const trait Into<T>: Sized {
    /// Converts this type into the (usually inferred) input type.
    fn into(self) -> T;
}

pub const trait From<T>: Sized {
    /// Converts to this type from the input type.
    fn from(value: T) -> Self;
}

impl<T, U> Into<U> for T
where
    U: From<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

pub mod std {
    pub mod ops {
        // Arithmetic operation lang items
        #[lang = "mul"]
        pub trait Mul<RHS = Self> {
            type Output;
            fn mul(self, rhs: RHS) -> Self::Output;
        }

        impl Mul for i32 {
            type Output = i32;
            fn mul(self, rhs: i32) -> Self::Output {
                0
            }
        }

        impl Mul for i64 {
            type Output = i64;
            fn mul(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "add"]
        pub trait Add<RHS = Self> {
            type Output;
            fn add(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i32 {
            type Output = i32;
            fn add(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u32 {
            type Output = u32;
            fn add(self, rhs: u32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for u64 {
            type Output = u64;
            fn add(self, rhs: u64) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for i64 {
            type Output = i64;
            fn add(self, rhs: i64) -> Self::Output {
                0
            }
        }

        #[lang = "sub"]
        pub trait Sub<RHS = Self> {
            type Output;
            fn sub(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for i32 {
            type Output = i32;
            fn sub(self, rhs: i32) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for u32 {
            type Output = u32;
            fn sub(self, rhs: u32) -> Self::Output {
                0
            }
        }

        #[lang = "div"]
        pub trait Div<RHS = Self> {
            type Output;
            fn div(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Div for i32 {
            type Output = i32;
            fn div(self, rhs: i32) -> Self::Output {
                0
            }
        }

        #[lang = "rem"]
        pub trait Rem<RHS = Self> {
            type Output;
            fn rem(self, rhs: RHS) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Rem for i32 {
            type Output = i32;
            fn rem(self, rhs: i32) -> Self::Output {
                0
            }
        }
    }
}
