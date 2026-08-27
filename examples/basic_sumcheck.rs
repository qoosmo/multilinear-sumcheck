use ark_bn254::Fr;
use multilinear_sumcheck::poly::CanonicalPoly;
use multilinear_sumcheck::sumcheck::{CanonicalProver, Verifier};

fn main() {
    let f = CanonicalPoly::new((1u64..=8).map(Fr::from).collect());
    let challenges = [Fr::from(3u64), Fr::from(7u64), Fr::from(11u64)];

    let prover = CanonicalProver::new(&f);
    let proof = prover.prove(&challenges);
    let oracle_eval = f.eval_circuit(&challenges);

    Verifier::verify(&proof, &challenges, oracle_eval).expect("valid Sumcheck proof must verify");

    println!(
        "verified {}-round Sumcheck proof ({} field elements)",
        proof.num_vars(),
        proof.size_in_field_elements()
    );
}
