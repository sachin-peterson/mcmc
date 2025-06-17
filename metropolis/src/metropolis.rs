use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use rand::thread_rng;

// Target distribution, e.g. standard normal N(0, 1)
pub fn target_distribution(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

// Metropolis algorithm:
// Inputs:
// x0           – starting point             
// n            – number of steps
// proposal_std – standard deviation for proposal distribution

// Outputs:
// Vec<f64>     - chain of samples
pub fn metropolis(x0: f64, n: usize, proposal_std: f64) -> Vec<f64> {

    // Random number generator
    let mut rng = thread_rng();

    // Uniform distribution for acceptance/rejection step
    let uniform = Uniform::new(0.0, 1.0);

    // Symmetric proposal distribution: N(0, std)
    let proposal = Normal::new(0.0, proposal_std).unwrap();

    // Current state
    let mut x = x0;

    // Allocate memory for Markov chain
    let mut chain = Vec::with_capacity(n);
    chain.push(x);

    // Metropolis loop
    for _ in 0..n {
        // Propose a new point x' = x + ε
        let epsilon = proposal.sample(&mut rng);
        let x_new = x + epsilon;

        // Evaluate target distribution @ x and x_new
        let pi_x = target_distribution(x);
        let pi_x_new = target_distribution(x_new);

        // Compute acceptance probability
        let alpha = (pi_x_new / pi_x).min(1.0);

        // Accept or reject the new point
        let u = uniform.sample(&mut rng);
        if u < alpha {x = x_new;}

        // Store current state
        chain.push(x);
    }

    // Return the Markov chain
    chain

}
