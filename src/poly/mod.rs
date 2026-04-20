//! Multilinear polynomial types.
//!
//! Two representations of the same mathematical object:
//!
//! - [`CanonicalPoly`] — coefficient vector in the monomial basis
//! - [`LagrangePoly`]  — evaluation vector over the Boolean hypercube
//!
//! Both implement [`MlPoly`], the common interface used by all higher layers.
//!
//! The conversion from [`CanonicalPoly`] to [`LagrangePoly`] is performed by
//! the circuit decomposition in [`crate::circuit`].

mod canonical;
mod lagrange;
mod traits;
mod uni;

pub use canonical::{CanonicalPoly, CanonicalTerm};
pub use lagrange::LagrangePoly;
pub use traits::MlPoly;
pub use uni::{UniDecomp, UniPoly, UniTerm};




