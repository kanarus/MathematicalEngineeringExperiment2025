#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod matrix;
mod plot;

pub use matrix::{Matrix, Vector};
pub use plot::Plotter;

pub const EPSILON: f64 = 1e-10;

/// Solve Ay = b by forward substitution:
/// 
/// ```text
/// y_i = (b_i - sum_{j=0}^{i-1} a_{ij} y_j) / a_{ii}
/// for i = 0, 1, ..., N-1
/// ```
pub fn forward_substitution<const N: usize>(
    lower_triangular_matrix: &Matrix<N, N>,
    b: &Vector<N>,
) -> Vector<N> {
    assert!(
        (0..N).all(|i| lower_triangular_matrix.column(i).take(i).all(|x| x.abs() < EPSILON)),
        "Matrix is not lower triangular"
    );
    
    let mut y = Vector::<N>::zeroed();
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lower_triangular_matrix[(i, j)] * y[j];
        }
        y[i] = (b[i] - sum) / lower_triangular_matrix[(i, i)];
    }
    y
}

/// Solve Ax = b by back substitution:
/// 
/// ```text
/// x_i = (b_i - sum_{j=i+1}^{n} a_{ij} x_j) / a_{ii}
/// for i = N-1, N-2, ..., 0
/// ```
pub fn back_substitution<const N: usize>(
    upper_triangular_matrix: &Matrix<N, N>,
    b: &Vector<N>,
) -> Vector<N> {
    assert!(
        (0..N).all(|i| upper_triangular_matrix.column(i).skip(i + 1).all(|x| x.abs() < EPSILON)),
        "Matrix is not upper triangular"
    );
    
    let mut x = Vector::<N>::zeroed();
    for i in (0..N).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..N {
            sum += upper_triangular_matrix[(i, j)] * x[j];
        }
        x[i] = (b[i] - sum) / upper_triangular_matrix[(i, i)];
    }
    x
}

fn random_value() -> f64 {
    use rand::{Rng, rng};
    rng().random_range(-1.0..=1.0)
}

fn with_elapsed<F, R>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let t = std::time::Instant::now();
    let result = f();
    let elapsed = t.elapsed();
    (result, elapsed)
}

pub struct EquationSolver<const N: usize> {
    f: fn(&Matrix<N, N>, &Vector<N>) -> Vector<N>,
}

#[derive(Debug)]
pub struct EquationExperimentStat<const N: usize> {
    pub solution: Vector<N>,
    pub elapsed: std::time::Duration,
    pub reference_solution: Vector<N>,
    pub residual_norm: f64,
    pub relative_error: f64,
}

impl<const N: usize> EquationSolver<N> {
    /// `f: (A, b) -> x` should solve the equation `Ax = b`
    pub fn new(
        f: fn(&Matrix<N, N>, &Vector<N>) -> Vector<N>,
    ) -> Self {
        Self { f }
    }
    
    /// A reference implementation for solving the equation `Ax = b`
    /// using nalgebra's LU decomposition.
    fn new_reference() -> Self {
        Self {
            f: |a: &Matrix<N, N>, b: &Vector<N>| -> Vector<N> {
                let view = nalgebra::DMatrix::from_fn(N, N, |i, j| a[(i, j)])
                    .lu()
                    .solve(&nalgebra::DVector::from_column_slice(b.as_ref()))
                    .unwrap();
                Vector::<N>::try_from(view.as_slice()).unwrap()
            }
        }
    }
    
    pub fn solve(&self, a: &Matrix<N, N>, b: &Vector<N>) -> Vector<N> {
        (self.f)(a, b)
    }

    pub fn experiment_randomly(&self) -> EquationExperimentStat<N> {
        let a = Matrix::<N, N>::from_fn(|_, _| random_value());
        let b = Vector::<N>::from_fn(|_, _| random_value());    
        
        let (solution, elapsed) = with_elapsed(|| self.solve(&a, &b));
        let reference_solution = Self::new_reference().solve(&a, &b);
        
        let residual_norm = (b - a * &solution).norm();
        let relative_error = (&solution - &reference_solution).norm() / reference_solution.norm();
        
        EquationExperimentStat {
            solution,
            reference_solution,
            elapsed,
            residual_norm,
            relative_error,
        }
    }
}

pub struct DominantEigenvalueSolver<const N: usize> {
    f: fn(&Matrix<N, N>) -> DominantEigenvalueSolution<N>,
}

#[derive(Debug)]
pub struct DominantEigenvalueSolution<const N: usize> {
    pub eigenvalue: f64,
    pub eigenvector: Vector<N>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub struct DominantEigenvalueExperimentStat<const N: usize> {
    pub solution: (f64, Vector<N>),
    pub iteration_count: usize,
    pub elapsed: std::time::Duration,
    pub reference_solution: (f64, Vector<N>),
    pub residual_norm: f64,
    pub eigenvalue_relative_error: f64,
    pub eigenvector_relative_error: f64,
}

impl<const N: usize> DominantEigenvalueSolver<N> {
    /// `f: A -> (λ, x)` should find the first eigenvalue λ and its eigenvector x of A
    pub fn new(f: fn(&Matrix<N, N>) -> DominantEigenvalueSolution<N>) -> Self {
        Self { f }
    }
    
    fn new_reference() -> Self {
        Self {
            f: |a: &Matrix<N, N>| -> DominantEigenvalueSolution<N> {
                let svd = nalgebra::DMatrix::from_fn(N, N, |i, j| a[(i, j)])
                    .svd(true, true);
                let largest_singular_value = svd
                    .singular_values
                    .get(0)
                    .expect("Matrix is singular")
                    .to_owned();
                let its_singular_vector = Vector::<N>::try_from(
                    svd.u
                        .unwrap()
                        .column(0)
                        .as_slice()
                ).unwrap();
                DominantEigenvalueSolution {
                    eigenvalue: largest_singular_value,
                    eigenvector: its_singular_vector,
                    iteration_count: 0,
                }
            }
        }
    }
    
    pub fn solve(&self, a: &Matrix<N, N>) -> DominantEigenvalueSolution<N> {
        (self.f)(a)
    }
    
    pub fn experiment_randomly(&self) -> DominantEigenvalueExperimentStat<N> {
        let a = {
            let random = Matrix::<N, N>::from_fn(|_, _| random_value());
            &random + random.transpose() // generate a symmetric matrix to ensure real eigenvalues
        };
        
        let (DominantEigenvalueSolution {
            eigenvalue,
            eigenvector,
            iteration_count,
        }, elapsed) = with_elapsed(|| self.solve(&a));
        
        let (reference_eigenvalue, reference_eigenvector) = {
            let r = Self::new_reference().solve(&a);
            (
                r.eigenvalue * eigenvalue.signum(),
                (&r.eigenvector) * (r.eigenvector.dot(&eigenvector).signum()),
            )
        };
        
        let residual_norm = (eigenvalue * &eigenvector - a * &eigenvector).norm();
        let eigenvalue_relative_error = (eigenvalue - reference_eigenvalue).abs() / reference_eigenvalue.abs();
        let eigenvector_relative_error = (&eigenvector - &reference_eigenvector).norm() / reference_eigenvector.norm();
        
        DominantEigenvalueExperimentStat {
            solution: (eigenvalue, eigenvector),
            iteration_count,
            elapsed,
            reference_solution: (reference_eigenvalue, reference_eigenvector),
            residual_norm,
            eigenvalue_relative_error,
            eigenvector_relative_error,
        }
    }
}

pub struct AllEigenvaluesSolver<const N: usize> {
    f: fn(&Matrix<N, N>) -> AllEigenvaluesSolution<N>,
}

#[derive(Debug)]
pub struct AllEigenvaluesSolution<const N: usize> {
    pub eigenvalues: Vector<N>,
    pub eigenvectors: Matrix<N, N>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub struct AllEigenvaluesExperimentStat<const N: usize> {
    pub solution: (Vector<N>, Matrix<N, N>),
    pub iteration_count: usize,
    pub elapsed: std::time::Duration,
    pub reference_solution: (Vector<N>, Matrix<N, N>),
    pub max_eigenvalue_residual_norm: f64,
    pub max_eigenvalues_relative_error: f64,
}
