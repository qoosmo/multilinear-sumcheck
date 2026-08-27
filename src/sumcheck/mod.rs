//! Sumcheck protocol — proof types, prover, and verifier.

pub mod proof;
pub mod prover;
pub mod verifier;

pub use proof::{RoundPoly, SumcheckProof};
pub use prover::{CanonicalProver, LagrangeProver};
pub use verifier::{Verifier, VerifierError};
