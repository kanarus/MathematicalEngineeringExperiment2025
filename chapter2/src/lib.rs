use nalgebra::{SMatrix, SVector};

pub const EPSILON: f64 = 1e-10;

/// Solve Ay = b by forward substitution:
/// 
/// ```text
/// y_i = (b_i - sum_{j=0}^{i-1} a_{ij} y_j) / a_{ii}
/// for i = 0, 1, ..., N-1
/// ```
pub fn forward_substitution<const N: usize>(
    lower_triangular_matrix: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> SVector<f64, N> {
    assert!(
        (0..N).all(|i| lower_triangular_matrix.column(i).iter().take(i).enumerate().all(|(j, x)| {dbg!(i, j); dbg!(x.abs()) < EPSILON})),
        "Matrix is not lower triangular"
    );
    
    let mut y = SVector::<f64, N>::zeros();
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
    upper_triangular_matrix: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> SVector<f64, N> {
    assert!(
        (0..N).all(|i| upper_triangular_matrix.column(i).iter().skip(i + 1).all(|x| x.abs() < EPSILON)),
        "Matrix is not upper triangular"
    );
    
    let mut x = SVector::<f64, N>::zeros();
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

#[derive(Debug)]
pub struct ExperimentResult<Solution, Errors> {
    pub solution: Solution,
    pub reference_solution: Solution,
    pub elapsed: std::time::Duration,
    pub errors: Errors,
}

pub struct EquationSolver<const N: usize> {
    f: fn(SMatrix<f64, N, N>, SVector<f64, N>) -> SVector<f64, N>,
}

#[derive(Debug)]
pub struct EquationErrors {
    pub residual_norm: f64,
    pub relative_error: f64,
}

impl<const N: usize> EquationSolver<N> {
    /// `f: (A, b) -> x` should solve the equation `Ax = b`
    pub fn new(
        f: fn(SMatrix<f64, N, N>, SVector<f64, N>) -> SVector<f64, N>,
    ) -> Self {
        Self { f }
    }
    
    pub fn experiment_randomly(&self) -> ExperimentResult<SVector<f64, N>, EquationErrors> {
        let a = SMatrix::<f64, N, N>::from_fn(|_, _| random_value());
        let b = SVector::<f64, N>::from_fn(|_, _| random_value());    
        
        let (solution, elapsed) = with_elapsed(|| (self.f)(a, b));
        
        let reference_solution = {
            let view = nalgebra::DMatrix::from_column_slice(N, N, a.as_slice())
                .lu()
                .solve(&b)
                .unwrap();
            SVector::<f64, N>::from_column_slice(view.as_slice())
        };
        
        let errors = EquationErrors {
            residual_norm: (b - a * &solution).norm(),
            relative_error: ( &solution - &reference_solution).norm() / reference_solution.norm(),
        };
        
        ExperimentResult {
            solution,
            reference_solution,
            elapsed,
            errors,
        }
    }
}

pub struct FirstEigenvalueSolver<const N: usize> {
    f: fn(SMatrix<f64, N, N>) -> (f64, SVector<f64, N>),
}

#[derive(Debug)]
pub struct EigenvalueErrors {
    pub eigenvalue_residual_norm: f64,
    pub eigenvalue_relative_error: f64,
    pub eigenvector_relative_error: f64,
}

impl<const N: usize> FirstEigenvalueSolver<N> {
    /// `f: A -> (λ, x)` should find the first eigenvalue λ and its eigenvector x of A
    pub fn new(f: fn(SMatrix<f64, N, N>) -> (f64, SVector<f64, N>)) -> Self {
        Self { f }
    }
    
    pub fn experiment_randomly(&self) -> ExperimentResult<(f64, SVector<f64, N>), EigenvalueErrors> {
        let a = SMatrix::<f64, N, N>::from_fn(|_, _| random_value());
        
        let ((eigenvalue, eigenvector), elapsed) = with_elapsed(|| (self.f)(a));
        
        let (reference_eigenvalue, reference_eigenvector) = {
            let first_eigenvalue = nalgebra::DMatrix::from_column_slice(N, N, a.as_slice())
                .eigenvalues()
                .expect("A is not diagonalizable")
                .get(0)
                .expect("No eigenvalues found");
            todo!()
        };
        
        let errors = todo!();
        
        ExperimentResult {
            solution: (eigenvalue, eigenvector),
            reference_solution: (reference_eigenvalue, reference_eigenvector),
            elapsed,
            errors,
        }
    }
}
