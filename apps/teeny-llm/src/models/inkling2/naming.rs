//! Inkling-2 (`Inkling2LoopForConditionalGeneration`), starting with Thinking
//! Machines' Inkling-2-Mini (`thinkingmachines/Inkling-2-Mini`): a 26.4B multimodal
//! mixture-of-experts model whose text path runs here on RISC-V.
//!
//! # The model
//!
//! 24 decoder layers, of which the middle twelve (`loop_config.core_start ..
//! core_end`, 6..18) run **three times over the same weights**, so a token costs
//! **48 layer executions from 24 sets of weights**. Each layer's feed-forward is a
//! sparse MoE: a router picks **8 of 128 experts** per token, and only those run.
//! 3.4B parameters are active per token, about 5.4B counting the extra passes.
//!
//! Attention is hybrid. Twenty layers are sliding-window; layers 5, 11, 17 and 23 are
//! full-attention and differ structurally -- separate q/k/v input norms instead of
//! one `attn_norm`, convolutions whose taps are projected per token, and three
//! attention-residual queries instead of one. There is no RoPE: position enters
//! Dotted node names for Inkling-2's graph builders.
//!
//! The builders are deep and many-armed, so the names come from a thread-local stack of
//! scope segments rather than a [`Scope`](crate::models::scope) threaded through every
//! call. Every node recorded with [`RecordNamed::record_named`] while a guard is alive is
//! named after the whole stack (`attn.wq_du`), which the runner then maps onto checkpoint
//! tensors. Recording outside any scope leaves the node unnamed.

use std::cell::RefCell;

use teeny_core::graph::{CustomData, DtypeRepr, SymTensor};

thread_local! {
    static SCOPE: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Pops its segment off the scope stack when dropped.
#[must_use = "the scope ends when the guard is dropped"]
pub struct NameGuard(());

impl Drop for NameGuard {
    fn drop(&mut self) {
        SCOPE.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

/// Pushes `segment` onto this thread's scope stack until the guard drops.
pub fn name_scope(segment: impl Into<String>) -> NameGuard {
    SCOPE.with(|s| s.borrow_mut().push(segment.into()));
    NameGuard(())
}

/// The current dotted scope, or `None` outside every scope.
pub fn current_scope() -> Option<String> {
    SCOPE.with(|s| {
        let s = s.borrow();
        (!s.is_empty()).then(|| s.join("."))
    })
}

/// [`SymTensor::record_custom`], naming the new node after the current scope.
pub trait RecordNamed {
    fn record_named(
        &self,
        data: CustomData,
        other_inputs: &[&SymTensor],
        dtype: Option<DtypeRepr>,
    ) -> SymTensor;
}

impl RecordNamed for SymTensor {
    fn record_named(
        &self,
        data: CustomData,
        other_inputs: &[&SymTensor],
        dtype: Option<DtypeRepr>,
    ) -> SymTensor {
        let node = self.record_custom(data, other_inputs, dtype);
        match current_scope() {
            Some(name) => node.named(name),
            None => node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scopes_nest_and_pop() {
        assert_eq!(current_scope(), None);
        let _a = name_scope("attn");
        {
            let _b = name_scope("wq_du");
            assert_eq!(current_scope().as_deref(), Some("attn.wq_du"));
        }
        assert_eq!(current_scope().as_deref(), Some("attn"));
    }
}
