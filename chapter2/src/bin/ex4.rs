use chapter2::{EPSILON, FirstEigenvalueSolver, EigenvalueErrors};
use nalgebra::{SMatrix, SVector};

fn solve_by_power_iteration<const N: usize>(a: SMatrix<f64, N, N>) -> (f64, SVector<f64, N>) {
    todo!()
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
