use rand::prelude::*;
use rand_distr::StandardNormal;
use rand::distributions::{Uniform};


// Log probability = –0.5 * q^2 
// Std normal, 1D
pub fn log_prob(q: f64) -> f64 {
    -0.5 * q * q
}

// Gradient of log probability = -q 
// Std normal, 1D
pub fn log_prob_grad(q: f64) -> f64 {
    -q
}

// Leapfrog algorithm
// 
// Inputs:
// q        – current position
// p        – current momentum (from std normal)
// epsilon  – step size for integrator
// l        – number of leapforg steps
// 
// Outputs:
// q_new    – proposed position
// p_new    – proposed momentum
pub fn leapfrog(q: f64, p: f64, epsilon: f64, l: usize) -> (f64, f64) {
    
    // Simulating from current state
    let mut q_new = q;

    // Half-step update of momentum
    let mut p_new = p + 0.5 * epsilon * log_prob_grad(q_new);

    // Updating position and momentum
    for i in 0..l {
        
        // Updating position
        q_new += epsilon * p_new;

        // Updating momentum
        if i != l - 1 {p_new += epsilon * log_prob_grad(q_new);}
    }

    // Half-step update of momentum
    p_new += 0.5 * epsilon * log_prob_grad(q_new);

    // Negate momentum
    p_new = -p_new;

    // Return proposed state
    (q_new, p_new)

}

// HMC algorithm – Computes the Hamiltonian H(q,p) = U(q) + K(p)
// 
// Inputs:
// q0       - starting position of Markov chain
// n        – number of samples
// epsilon  – step size for integrator
// l        – number of leapforg steps
// 
// Outputs:
// Vec<f64> – chain of samples
pub fn hamilton_monte_carlo(q0: f64, n: usize, epsilon: f64, l: usize) -> Vec<f64> {

    // Random number generator
    let mut rng = rand::thread_rng();

    // Create uniform distribution U(0,1)
    let uniform = Uniform::new(0.0, 1.0);

    // Preallocate space for chain
    let mut chain = Vec::with_capacity(n);

    // Current position
    let mut q = q0;

    // HMC loop
    for _ in 0..n {
        
        // Generate initial momentum p ~ N(0,1)
        let p: f64 = rng.sample(StandardNormal);

        // Compute current Hamiltonian energy
        let current_log = log_prob(q);
        let current_ham = -current_log + 0.5 * p * p;

        // Simulate Hamiltonian dynamics
        let (q_new, p_new) = leapfrog(q, p, epsilon, l);

        // Compute proposed Hamiltonian
        let proposed_log = log_prob(q_new);
        let proposed_ham = -proposed_log + 0.5 * p_new * p_new;

        // Metropolis acceptance step
        let alpha: f64 = ((current_ham - proposed_ham).exp()).min(1.0);
        let u = uniform.sample(&mut rng);
        if u < alpha {q = q_new;}

        // Update Markov chain
        chain.push(q);
    }

    // Return Markov chain
    chain

}
