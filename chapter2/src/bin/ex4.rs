use chapter2::{EPSILON, FirstEigenvalueSolver, EigenvalueErrors};
use nalgebra::{SMatrix, SVector};

fn solve_by_power_iteration<const N: usize>(a: SMatrix<f64, N, N>) -> (f64, SVector<f64, N>) {
    const MAX_ITERATIONS: usize = 100_000;
    
    let mut mu = Vec::<f64>::new();
    let mut x_k = SVector::<f64, N>::from_fn(|_, _| 1.0);
    for _ in 0..MAX_ITERATIONS {
        let y_k = a * x_k;
        
        let (i, _max_abs) = x_k
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| f64::partial_cmp(&p.abs(), &q.abs()).expect("found NaN or Inf"))
            .expect("Vector is zero");
        
        let mu_k = y_k[i] / x_k[i];
        if mu.last().is_some_and(|mu_prev| (mu_k - mu_prev).abs() < EPSILON) {
            return (mu_k, x_k);
        }
        
        x_k = y_k / y_k.norm();
        mu.push(mu_k);
    }
    
    panic!("`mu` seems to diverge");
}

fn experiment<const N: usize>() -> chapter2::ExperimentResult<(f64, SVector<f64, N>), EigenvalueErrors> {
    FirstEigenvalueSolver::new(solve_by_power_iteration).experiment_randomly()
}

fn main() {
    for _ in 0..100 {
        dbg!(experiment::<100>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
