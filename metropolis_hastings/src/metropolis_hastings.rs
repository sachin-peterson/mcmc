use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use statrs::distribution::{Normal, Continuous};


// Target distribution, e.g. standard normal N(0, 1)
pub fn target_distribution(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

// Metropolis-Hastings algorithm:
// Inputs:
// x0           – starting point             
// n            – number of steps
// mean_shift   – mean shift for asymmetric proposal
// proposal_std – standard deviation for proposal distribution

// Outputs:
// Vec<f64>     - chain of samples
pub fn metropolis_hastings(x0: f64, n: usize, mean_shift: f64, proposal_std: f64) -> Vec<f64> {
    
    // Random number generator
    let mut rng = thread_rng();

    // Uniform distribution for acceptance/rejection step
    let uniform = Uniform::new(0.0, 1.0);

    // Current state
    let mut x = x0;

    // Allocate memory for Markov chain
    let mut chain = Vec::with_capacity(n);
    chain.push(x);

    // Metropolis-Hastings loop
    for _ in 0..n {
        // Propose a new point x' ~ N(x + mean_shift, proposal_std^2)
        let proposal = Normal::new(x + mean_shift, proposal_std).unwrap();
        let x_new = proposal.sample(&mut rng);

        // Evaluate target distribution @ x and x_new
        let pi_x = target_distribution(x);
        let pi_x_new = target_distribution(x_new);

        // Evaluate internal proposal densities
        let q_forward = Normal::new(x + mean_shift, proposal_std).unwrap().pdf(x_new);
        let q_backward = Normal::new(x_new + mean_shift, proposal_std).unwrap().pdf(x);

        // Compute acceptance probability
        let alpha = (pi_x_new * q_backward / (pi_x * q_forward)).min(1.0);

        // Accept or reject the new point
        let u = uniform.sample(&mut rng);
        if u < alpha {x = x_new;}

        // Store current state
        chain.push(x);
    }

    // Return the Markov chain
    chain

}
