use chapter2::{Matrix, Vector};
use chapter2::{EPSILON, DominantEigenvalueSolver, DominantEigenvalueSolution};

fn solve_by_power_iteration<const N: usize>(a: &Matrix<N, N>) -> DominantEigenvalueSolution<N> {
    const MAX_ITERATIONS: usize = 100_000;
    
    let mut mu = Vec::<f64>::new();
    let mut x_k = Vector::<N>::filled_with(1.0);
    for count in 1..MAX_ITERATIONS {
        let y_k = a * &x_k;
        
        let (i, _max_abs) = x_k
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| f64::partial_cmp(&p.abs(), &q.abs()).expect("found NaN or Inf"))
            .expect("Vector is zero");
        
        let mu_k = y_k[i] / x_k[i];
        if mu.last().is_some_and(|it| (it.abs() - mu_k.abs()).abs() < EPSILON) {
            return DominantEigenvalueSolution {
                eigenvalue: mu_k,
                eigenvector: x_k,
                iteration_count: count,
            };
        }
        
        x_k = y_k.normalized();
        mu.push(mu_k);
    }
    
    panic!("`mu` seems to diverge");
}

fn main() {
    let solver = DominantEigenvalueSolver::new(solve_by_power_iteration::<100>);
    
    for _ in 0..100 {
        dbg!(solver.experiment_randomly());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solve_by_power_iteration() {
        let a = Matrix::<3, 3>::from([
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]);
        
        let solution = dbg!(solve_by_power_iteration(&a));
        
        assert!((solution.eigenvalue - (f64::sqrt(2.) + 2.)).abs() < EPSILON);
        assert!((solution.eigenvector.normalized() - Vector::<3>::from([
            1., f64::sqrt(2.), 1.
        ]).normalized()).norm() < EPSILON);
    }
}
