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

/// A reference implementation for solving the equation `Ax = b`
/// using nalgebra's LU decomposition.
fn reference_equation_solver<const N: usize>(
    a: SMatrix<f64, N, N>,
    b: SVector<f64, N>,
) -> SVector<f64, N> {
    let view = nalgebra::DMatrix::from_column_slice(N, N, a.as_slice())
        .lu()
        .solve(&b)
        .unwrap();
    SVector::<f64, N>::from_column_slice(view.as_slice())
}

pub struct EquationSolver<const N: usize> {
    f: fn(SMatrix<f64, N, N>, SVector<f64, N>) -> SVector<f64, N>,
}

#[derive(Debug)]
pub struct EquationExperimentStat<const N: usize> {
    pub solution: SVector<f64, N>,
    pub elapsed: std::time::Duration,
    pub reference_solution: SVector<f64, N>,
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
    
    pub fn experiment_randomly(&self) -> EquationExperimentStat<N> {
        let a = SMatrix::<f64, N, N>::from_fn(|_, _| random_value());
        let b = SVector::<f64, N>::from_fn(|_, _| random_value());    
        
        let (solution, elapsed) = with_elapsed(|| (self.f)(a, b));
        
        let reference_solution = reference_equation_solver(a, b);
        
        EquationExperimentStat {
            solution,
            reference_solution,
            elapsed,
            residual_norm: (b - a * &solution).norm(),
            relative_error: ( &solution - &reference_solution).norm() / reference_solution.norm(),
        }
    }
}

pub struct DominantEigenvalueSolver<const N: usize> {
    f: fn(SMatrix<f64, N, N>) -> DominantEigenvalueSolution<N>,
}

#[derive(Debug)]
pub struct DominantEigenvalueSolution<const N: usize> {
    pub eigenvalue: f64,
    pub eigenvector: SVector<f64, N>,
    pub iteration_count: usize,
}

#[derive(Debug)]
pub struct DominantEigenvalueExperimentStat<const N: usize> {
    pub solution: (f64, SVector<f64, N>),
    pub iteration_count: usize,
    pub elapsed: std::time::Duration,
    pub reference_solution: (f64, SVector<f64, N>),
    pub eigenvalue_residual_norm: f64,
    pub eigenvalue_relative_error: f64,
    pub eigenvector_relative_error: f64,
}

impl<const N: usize> DominantEigenvalueSolver<N> {
    /// `f: A -> (λ, x)` should find the first eigenvalue λ and its eigenvector x of A
    pub fn new(f: fn(SMatrix<f64, N, N>) -> DominantEigenvalueSolution<N>) -> Self {
        Self { f }
    }
    
    pub fn experiment_randomly(&self) -> DominantEigenvalueExperimentStat<N> {
        let a = {
            let random = SMatrix::<f64, N, N>::from_fn(|_, _| random_value());
            random + random.transpose() // generate a symmetric matrix to ensure real eigenvalues
        };
        
        let (DominantEigenvalueSolution {
            eigenvalue,
            eigenvector,
            iteration_count,
        }, elapsed) = with_elapsed(|| (self.f)(a));
        
        let (reference_eigenvalue, reference_eigenvector) = {
            let svd = nalgebra::DMatrix::from_column_slice(N, N, a.as_slice())
                .svd(true, true);
            let largest_singular_value = svd
                .singular_values
                .get(0)
                .expect("Matrix is singular")
                .to_owned();
            let its_singular_vector = SVector::<_, N>::from_column_slice(&svd
                .u
                .unwrap()
                .column(0)
                .as_slice()
            );
            (
                largest_singular_value * eigenvalue.signum(),
                its_singular_vector * (its_singular_vector.dot(&eigenvector).signum()),
            )
        };
        
        DominantEigenvalueExperimentStat {
            solution: (eigenvalue, eigenvector),
            iteration_count,
            elapsed,
            reference_solution: (reference_eigenvalue, reference_eigenvector),
            eigenvalue_residual_norm: (eigenvalue * eigenvector - a * eigenvector).norm(),
            eigenvalue_relative_error: (eigenvalue - reference_eigenvalue).abs() / reference_eigenvalue.abs(),
            eigenvector_relative_error: (eigenvector - reference_eigenvector).norm() / reference_eigenvector.norm(),
        }
    }
}
