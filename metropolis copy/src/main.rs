mod metropolis; use metropolis::metropolis;

use rayon::prelude::*;


fn main() {
    // Input parameters for Metropolis
    let x0 = 0.0;
    let n = 1_000_000;
    let proposal_std = 1.0;

    // Number of Markov chains
    let m = 4;

    // Tolerance for Rhat
    let convergence_threshold = 1e-3;

    // Run independent Metropolis chains in parallel
    let chains: Vec<Vec<f64>> = (0..m).into_par_iter()
        .map(|_| metropolis(x0, n, proposal_std))
        .collect();

    // Check R-hat at increasing sample sizes
    for t in (1000..n).step_by(1000) {
        let r_hat = compute_rhat_at_step(&chains, t);
        println!("R-hat at t = {}: {:.6}", t, r_hat);
        if (r_hat - 1.0).abs() < convergence_threshold {
            println!("\nConvergence detected at t = {} with R-hat ≈ {:.6}", t, r_hat);
            break;
        }
    }

    // Flatten chains into a single vector
    let all_samples: Vec<f64> = chains.iter().flatten().cloned().collect();

    // Overall mean and variance
    let overall_mean = compute_mean(&all_samples);
    let overall_var = compute_var(&all_samples, overall_mean);

    // Printing results
    println!("\nTrue mean: 0.0");
    println!("True variance: 1.0");

    println!("\nOverall mean: {:.6}", overall_mean);
    println!("Overall variance: {:.6}", overall_var);

}

// Compute the Rhat statistic
fn compute_rhat_at_step(chains: &Vec<Vec<f64>>, t: usize) -> f64 {
    // Number of chains
    let m = chains.len();

    // Compute the mean of the first `t` samples in each chain
    let chain_means: Vec<f64> = chains
        .iter()
        .map(|chain| compute_mean(&chain[..t]))
        .collect();

    // Compute the overall mean
    let overall_mean: f64 = chain_means.iter().sum::<f64>() / m as f64;

    // Compute the between-chain variance B
    let b: f64 = (t as f64 / (m as f64 - 1.0)) *
        chain_means.iter()
            .map(|mean| (mean - overall_mean).powi(2))
            .sum::<f64>();

    // Compute the within-chain variance W
    let w: f64 = chains.iter()
        .zip(chain_means.iter())
        .map(|(chain, mean)| compute_var(&chain[..t], *mean))
        .sum::<f64>() / m as f64;

    // Estimate v_hat
    let v_hat = (1.0 - 1.0 / t as f64) * w + (1.0 / t as f64) * b;

    // Compute R-hat
    (v_hat / w).sqrt()
}

// Compute mean in parallel
fn compute_mean(chain: &[f64]) -> f64 {
    let n = chain.len() as f64;
    let sum: f64 = chain.par_iter().sum();
    sum / n
}

// Compute variance in parallel
fn compute_var(chain: &[f64], mean: f64) -> f64 {
    let n = chain.len() as f64;
    let sq_diffs: f64 = chain.par_iter().map(|x| (x - mean).powi(2)).sum();
    sq_diffs / (n - 1.0)
}
