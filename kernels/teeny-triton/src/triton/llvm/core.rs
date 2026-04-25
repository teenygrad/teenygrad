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
pub unsafe auto trait MetaSized {}

#[lang = "pointee_sized"]
pub unsafe auto trait PointeeSized {}

// Required language items for no_core
#[lang = "sized"]
pub trait Sized {}

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

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {}

// Enable &[T; N] → &[T] coercions without unsafe slice_from_raw_parts.
// The compiler automatically implements Unsize<[T]> for [T; N]; this impl
// wires up the syntactic coercion for shared references.
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}

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

#[lang = "panic_const_div_by_zero"]
pub fn panic_const_div_by_zero() -> ! {
    loop {}
}

#[lang = "panic_const_rem_overflow"]
pub fn panic_const_rem_overflow() -> ! {
    loop {}
}

#[lang = "panic_const_rem_by_zero"]
pub fn panic_const_rem_by_zero() -> ! {
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

impl Copy for f64 {}
impl Clone for f64 {
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
impl Copy for usize {}
impl Clone for usize {
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


#[lang = "eq"]
pub trait PartialEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool;
}

impl PartialEq for i32 {
    fn eq(&self, other: &i32) -> bool { false }
    fn ne(&self, other: &i32) -> bool { false }
}

#[lang = "partial_ord"]
pub trait PartialOrd<Rhs: ?Sized = Self> {
    fn lt(&self, other: &Rhs) -> bool;
    fn gt(&self, other: &Rhs) -> bool;
    fn le(&self, other: &Rhs) -> bool;
    fn ge(&self, other: &Rhs) -> bool;
}

impl PartialOrd for i32 {
    fn lt(&self, _other: &i32) -> bool { false }
    fn gt(&self, _other: &i32) -> bool { false }
    fn le(&self, _other: &i32) -> bool { false }
    fn ge(&self, _other: &i32) -> bool { false }
}

#[lang = "Option"]
pub enum Option<T> {
    #[lang = "None"]
    None,
    #[lang = "Some"]
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

impl<T> From<T> for T {
    fn from(value: T) -> Self {
        value
    }
}

impl<T, U> Into<U> for T
where
    U: From<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

pub mod iter {
    use super::PartialOrd;

    #[lang = "iterator"]
    pub trait Iterator {
        type Item;

        #[lang = "next"]
        fn next(&mut self) -> super::Option<Self::Item>;
    }

    pub trait IntoIterator {
        type Item;
        type IntoIter: Iterator<Item = Self::Item>;

        #[lang = "into_iter"]
        fn into_iter(self) -> Self::IntoIter;
    }

    impl Iterator for super::core::ops::Range<i32> {
        type Item = i32;

        fn next(&mut self) -> super::Option<Self::Item> {
            if self.start < self.end {
                let v = self.start;
                self.start = v + 1;
                super::Option::Some(v)
            } else {
                super::Option::None
            }
        }
    }

    impl IntoIterator for super::core::ops::Range<i32> {
        type Item = i32;
        type IntoIter = super::core::ops::Range<i32>;

        fn into_iter(self) -> Self::IntoIter {
            self
        }
    }
}

pub mod core {
    pub mod ops {
        #[lang = "Range"]
        pub struct Range<Idx> {
            pub start: Idx,
            pub end: Idx,
        }

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

        // Just a dummy, the compiler will generate the correct implementation
        impl Mul for f32 {
            type Output = f32;
            fn mul(self, rhs: f32) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Mul for f64 {
            type Output = f64;
            fn mul(self, rhs: f64) -> Self::Output {
                0.0
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

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for f32 {
            type Output = f32;
            fn add(self, rhs: f32) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Add for f64 {
            type Output = f64;
            fn add(self, rhs: f64) -> Self::Output {
                0.0
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

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for f32 {
            type Output = f32;
            fn sub(self, rhs: f32) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Sub for f64 {
            type Output = f64;
            fn sub(self, rhs: f64) -> Self::Output {
                0.0
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

        // Just a dummy, the compiler will generate the correct implementation
        impl Div for f32 {
            type Output = f32;
            fn div(self, rhs: f32) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Div for f64 {
            type Output = f64;
            fn div(self, rhs: f64) -> Self::Output {
                0.0
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

        #[lang = "neg"]
        pub trait Neg {
            type Output;
            fn neg(self) -> Self::Output;
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for i32 {
            type Output = i32;
            fn neg(self) -> Self::Output {
                0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for f32 {
            type Output = f32;
            fn neg(self) -> Self::Output {
                0.0
            }
        }

        // Just a dummy, the compiler will generate the correct implementation
        impl Neg for f64 {
            type Output = f64;
            fn neg(self) -> Self::Output {
                0.0
            }
        }

        #[lang = "bitand"]
        pub trait BitAnd<RHS = Self> {
            type Output;
            fn bitand(self, rhs: RHS) -> Self::Output;
        }

        impl BitAnd for bool {
            type Output = bool;
            fn bitand(self, rhs: bool) -> Self::Output {
                false
            }
        }

        #[lang = "bitor"]
        pub trait BitOr<RHS = Self> {
            type Output;
            fn bitor(self, rhs: RHS) -> Self::Output;
        }

        impl BitOr for bool {
            type Output = bool;
            fn bitor(self, rhs: bool) -> Self::Output {
                false
            }
        }
    }
}
