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

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields};

pub fn jit_module_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Parse struct fields and collect module and tensor fields
    let mut module_fields = Vec::new();
    let mut tensor_fields = Vec::new();

    if let Data::Struct(data) = &input.data {
        if let Fields::Named(fields) = &data.fields {
            for field in &fields.named {
                if let Some(field_ident) = &field.ident {
                    if has_attribute(&field.attrs, "module") {
                        module_fields.push(field_ident);
                    } else if has_attribute(&field.attrs, "tensor") {
                        tensor_fields.push(field_ident);
                    }
                }
            }
        }
    }

    // Generate field transfer code
    let module_transfers: Vec<_> = module_fields
        .iter()
        .map(|field| {
            quote! {
                self.#field = self.#field.to_device(&device)?;
            }
        })
        .collect();

    let tensor_transfers: Vec<_> = tensor_fields
        .iter()
        .map(|field| {
            quote! {
                self.#field = self.#field.to_device(&device)?;
            }
        })
        .collect();

    // Generate JIT-related implementations
    let result = quote! {
        use std::sync::Arc;
        use teeny_core::device::Device;

        impl #impl_generics #name #ty_generics #where_clause {
            fn to_device(&mut self, device: Arc<dyn Device>) -> Result<(), Box<dyn std::error::Error>> {
                // Transfer modules to device
                #(#module_transfers)*

                // Transfer tensors to device
                #(#tensor_transfers)*

                Ok(())
            }
        }
    };

    result.into()
}

fn has_attribute(attrs: &[Attribute], attr_name: &str) -> bool {
    attrs.iter().any(|attr| {
        if attr.path().is_ident(attr_name) {
            return true;
        }

        // Also check for attributes with the same name in different forms
        attr.path()
            .segments
            .iter()
            .any(|segment| segment.ident == attr_name)
    })
}
