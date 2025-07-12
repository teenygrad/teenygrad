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
use syn::{
    Attribute, Block, Expr, FnArg, Generics, Ident, Pat, ReturnType, Stmt, Type, Visibility,
    WhereClause,
};

/// Represents a complete Rust function with optional type information
#[derive(Clone)]
pub struct Function {
    /// Function attributes (e.g., #[inline], #[no_mangle])
    pub attrs: Vec<Attribute>,
    /// Function visibility
    pub vis: Visibility,
    /// Function signature
    pub sig: FunctionSignature,
    /// Function body
    pub body: FunctionBody,
    /// Optional semantic information added by the analyzer
    pub semantic_info: Option<SemanticInfo>,
}

/// Function signature with parameters and return type
#[derive(Clone)]
pub struct FunctionSignature {
    /// Function name
    pub name: Ident,
    /// Generic parameters
    pub generics: Generics,
    /// Function parameters
    pub params: Vec<Parameter>,
    /// Return type
    pub return_type: ReturnType,
    /// Where clause for generic bounds
    pub where_clause: Option<WhereClause>,
}

/// Function parameter with optional type information
#[derive(Clone)]
pub struct Parameter {
    /// Parameter pattern (e.g., `x`, `mut y`, `(a, b)`)
    pub pat: Pat,
    /// Parameter type
    pub ty: Type,
    /// Optional semantic type information
    pub semantic_type: Option<SemanticType>,
    /// Whether parameter is mutable
    pub is_mut: bool,
    /// Whether parameter is a reference
    pub is_ref: bool,
}

/// Function body containing statements
#[derive(Clone)]
pub struct FunctionBody {
    /// Statements in the function body
    pub stmts: Vec<Statement>,
    /// Optional return expression
    pub return_expr: Option<Expression>,
}

/// Statement in the function body
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Statement {
    /// Local variable declaration (let binding)
    Local(LocalStatement),
    /// Expression statement
    Expr(ExpressionStatement),
    /// Item declaration (nested function, struct, etc.)
    Item(ItemStatement),
    /// Empty statement (semicolon)
    Empty,
}

/// Local variable declaration
#[derive(Clone)]
pub struct LocalStatement {
    /// Variable pattern
    pub pat: Pat,
    /// Variable type annotation (optional)
    pub ty: Option<Type>,
    /// Initializer expression
    pub init: Option<Expression>,
    /// Optional semantic type information
    pub semantic_type: Option<SemanticType>,
    /// Whether variable is mutable
    pub is_mut: bool,
}

/// Expression statement
#[derive(Clone)]
pub struct ExpressionStatement {
    /// The expression
    pub expr: Expression,
    /// Whether statement ends with semicolon
    pub has_semicolon: bool,
}

/// Item statement (nested declarations)
#[derive(Clone)]
pub struct ItemStatement {
    /// The item (function, struct, enum, etc.)
    pub item: syn::Item,
}

/// Expression with optional type information
#[derive(Clone)]
pub struct Expression {
    /// The underlying syn expression
    pub expr: Expr,
    /// Optional semantic type information
    pub semantic_type: Option<SemanticType>,
    /// Whether expression is mutable
    pub is_mut: bool,
    /// Whether expression is a reference
    pub is_ref: bool,
}

/// Semantic type information added by the analyzer
#[derive(Clone)]
pub struct SemanticType {
    /// The resolved type
    pub ty: Type,
    /// Type size in bytes (if known)
    pub size: Option<usize>,
    /// Whether type is sized
    pub is_sized: bool,
    /// Whether type implements Copy
    pub is_copy: bool,
    /// Whether type implements Clone
    pub is_clone: bool,
    /// Whether type implements Debug
    pub is_debug: bool,
    /// Generic type parameters (if any)
    pub generic_params: Vec<Type>,
    /// Lifetime parameters (if any)
    pub lifetimes: Vec<String>,
}

/// Semantic information for the entire function
#[derive(Clone)]
pub struct SemanticInfo {
    /// Type information for all variables
    pub variable_types: HashMap<String, SemanticType>,
    /// Type information for all expressions
    pub expression_types: HashMap<usize, SemanticType>,
    /// Control flow information
    pub control_flow: ControlFlowInfo,
    /// Memory usage information
    pub memory_info: MemoryInfo,
    /// Error information (if any)
    pub errors: Vec<SemanticError>,
}

/// Control flow information
#[derive(Debug, Clone)]
pub struct ControlFlowInfo {
    /// Whether function can panic
    pub can_panic: bool,
    /// Whether function can return early
    pub can_return_early: bool,
    /// Whether function has infinite loops
    pub has_infinite_loops: bool,
    /// Whether function has recursive calls
    pub has_recursion: bool,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Stack size estimate
    pub stack_size: Option<usize>,
    /// Whether function allocates on heap
    pub allocates_heap: bool,
    /// Whether function uses dynamic dispatch
    pub uses_dynamic_dispatch: bool,
}

/// Semantic analysis error
#[derive(Debug, Clone)]
pub struct SemanticError {
    /// Error kind
    pub kind: SemanticErrorKind,
    /// Error message
    pub message: String,
    /// Error location (line/column)
    pub location: Option<(usize, usize)>,
}

/// Types of semantic errors
#[derive(Debug, Clone)]
pub enum SemanticErrorKind {
    /// Type mismatch
    TypeMismatch,
    /// Undefined variable
    UndefinedVariable,
    /// Unused variable
    UnusedVariable,
    /// Borrow checker error
    BorrowError,
    /// Lifetime error
    LifetimeError,
    /// Generic error
    Generic,
}

/// Trait for converting syn types to our AST
pub trait FromSyn {
    type Output;
    fn from_syn(input: Self::Output) -> Self;
}

/// Trait for converting our AST back to syn types
pub trait ToSyn {
    type Output;
    fn to_syn(&self) -> Self::Output;
}

impl FromSyn for Function {
    type Output = syn::ItemFn;

    fn from_syn(item_fn: syn::ItemFn) -> Self {
        let syn::ItemFn {
            attrs,
            vis,
            sig,
            block,
            ..
        } = item_fn;

        // Convert function signature
        let sig = FunctionSignature {
            name: sig.ident,
            generics: sig.generics.clone(),
            params: sig
                .inputs
                .into_iter()
                .map(|input| {
                    match input {
                        FnArg::Typed(pat_type) => {
                            Parameter {
                                pat: *pat_type.pat,
                                ty: *pat_type.ty,
                                semantic_type: None,
                                is_mut: false, // TODO: extract from pattern
                                is_ref: false, // TODO: extract from type
                            }
                        }
                        FnArg::Receiver(_) => {
                            // Handle self parameter
                            Parameter {
                                pat: Pat::Ident(syn::PatIdent {
                                    attrs: vec![],
                                    by_ref: None,
                                    mutability: None,
                                    ident: Ident::new("self", proc_macro2::Span::call_site()),
                                    subpat: None,
                                }),
                                ty: Type::Path(syn::TypePath {
                                    qself: None,
                                    path: syn::Path::from(Ident::new(
                                        "Self",
                                        proc_macro2::Span::call_site(),
                                    )),
                                }),
                                semantic_type: None,
                                is_mut: false,
                                is_ref: false,
                            }
                        }
                    }
                })
                .collect(),
            return_type: sig.output,
            where_clause: sig.generics.where_clause,
        };

        // Convert function body
        let body = FunctionBody {
            stmts: block
                .stmts
                .into_iter()
                .map(|stmt| match stmt {
                    Stmt::Local(local) => Statement::Local(LocalStatement {
                        pat: local.pat,
                        ty: None, // AXM - Fixme
                        init: local.init.map(|expr| Expression {
                            expr: *expr.expr,
                            semantic_type: None,
                            is_mut: false,
                            is_ref: false,
                        }),
                        semantic_type: None,
                        is_mut: false, // AXM - Fixme
                    }),
                    Stmt::Expr(expr, semicolon) => Statement::Expr(ExpressionStatement {
                        expr: Expression {
                            expr,
                            semantic_type: None,
                            is_mut: false,
                            is_ref: false,
                        },
                        has_semicolon: semicolon.is_some(),
                    }),
                    Stmt::Item(item) => Statement::Item(ItemStatement { item }),
                    Stmt::Macro(macro_stmt) => Statement::Expr(ExpressionStatement {
                        expr: Expression {
                            expr: Expr::Macro(syn::ExprMacro {
                                attrs: macro_stmt.attrs,
                                mac: macro_stmt.mac,
                            }),
                            semantic_type: None,
                            is_mut: false,
                            is_ref: false,
                        },
                        has_semicolon: macro_stmt.semi_token.is_some(),
                    }),
                })
                .collect(),
            return_expr: None, // TODO: extract from last expression if it's not a statement
        };

        Function {
            attrs,
            vis,
            sig,
            body,
            semantic_info: None,
        }
    }
}

impl ToSyn for Function {
    type Output = syn::ItemFn;

    fn to_syn(&self) -> syn::ItemFn {
        // Convert back to syn::ItemFn
        // This is a simplified conversion - in practice you'd want more robust handling
        syn::ItemFn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: syn::Signature {
                constness: None,
                asyncness: None,
                unsafety: None,
                abi: None,
                fn_token: syn::token::Fn(proc_macro2::Span::call_site()),
                ident: self.sig.name.clone(),
                generics: self.sig.generics.clone(),
                paren_token: syn::token::Paren(proc_macro2::Span::call_site()),
                inputs: self
                    .sig
                    .params
                    .iter()
                    .map(|param| {
                        FnArg::Typed(syn::PatType {
                            attrs: vec![],
                            pat: Box::new(param.pat.clone()),
                            colon_token: syn::token::Colon(proc_macro2::Span::call_site()),
                            ty: Box::new(param.ty.clone()),
                        })
                    })
                    .collect(),
                variadic: None,
                output: self.sig.return_type.clone(),
            },
            block: Box::new(Block {
                brace_token: syn::token::Brace(proc_macro2::Span::call_site()),
                stmts: self
                    .body
                    .stmts
                    .iter()
                    .map(|stmt| match stmt {
                        Statement::Local(local) => Stmt::Local(syn::Local {
                            attrs: vec![],
                            let_token: syn::token::Let(proc_macro2::Span::call_site()),
                            pat: local.pat.clone(),
                            init: None, // AXM - Fixme
                            semi_token: syn::token::Semi(proc_macro2::Span::call_site()),
                        }),
                        Statement::Expr(expr_stmt) => Stmt::Expr(
                            expr_stmt.expr.expr.clone(),
                            if expr_stmt.has_semicolon {
                                Some(syn::token::Semi(proc_macro2::Span::call_site()))
                            } else {
                                None
                            },
                        ),
                        Statement::Item(item) => Stmt::Item(item.item.clone()),
                        Statement::Empty => Stmt::Expr(
                            syn::Expr::Verbatim(proc_macro2::TokenStream::new()),
                            Some(syn::token::Semi(proc_macro2::Span::call_site())),
                        ),
                    })
                    .collect(),
            }),
        }
    }
}

/// Builder for creating AST nodes
pub struct AstBuilder;

impl AstBuilder {
    /// Create a new function AST from a syn ItemFn
    pub fn function_from_syn(item_fn: syn::ItemFn) -> Function {
        Function::from_syn(item_fn)
    }

    /// Create a new semantic type
    pub fn semantic_type(ty: Type) -> SemanticType {
        SemanticType {
            ty,
            size: None,
            is_sized: true,
            is_copy: false,
            is_clone: false,
            is_debug: false,
            generic_params: vec![],
            lifetimes: vec![],
        }
    }

    /// Create a new semantic error
    pub fn semantic_error(kind: SemanticErrorKind, message: String) -> SemanticError {
        SemanticError {
            kind,
            message,
            location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_str;

    #[test]
    fn test_function_parsing() {
        let code = r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;

        let item_fn: syn::ItemFn = parse_str(code).unwrap();
        let function = AstBuilder::function_from_syn(item_fn);

        assert_eq!(function.sig.name.to_string(), "add");
        assert_eq!(function.sig.params.len(), 2);
        assert!(function.semantic_info.is_none());
    }
}
