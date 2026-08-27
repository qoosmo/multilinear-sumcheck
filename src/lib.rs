//! Basis-aware multilinear polynomial algorithms and the Sumcheck protocol.
//!
//! The crate provides canonical (coefficient) and Lagrange (Boolean-hypercube
//! evaluation) representations, tree/circuit decompositions, linear-time
//! evaluation kernels, basis-specific Sumcheck provers, and a stateless
//! verifier.
//!
//! This is a research implementation. It is not a complete SNARK/STARK and
//! does not currently include Fiat-Shamir, a polynomial commitment scheme, or
//! zero-knowledge masking.

#![forbid(unsafe_code)]

pub mod circuit;
pub mod poly;
pub mod sumcheck;
