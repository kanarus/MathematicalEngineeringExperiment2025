#import "@preview/codelst:2.0.2": sourcecode

#set page(
  paper: "a4",
)
#set text(
  font: "Noto Serif CJK JP",
  size: 10pt,
)
#set heading(numbering: "1.")
#show heading: set block(above: 2em, below: 1em)
#show link: set text(font: "Mint Mono")

#set align(center)

#title("数理工学実験「数値線形代数」レポート")

#v(1em)

(10/7, 14-15, 20 工学部総合校舎数理計算室にて実施)

#v(1em)

#set align(right)

３回生 佐々木哉人 (学生番号: 1029-34-3748)

2025年10月27日

#set align(start)

= はじめに <intro>

本レポートは, 実験資料第２章「数値線形代数」 @TextBookChaptor2 の課題１〜５の結果を報告するものである.

用いた実験環境は以下の通りである:

#align(center)[
  #table(
    columns: (auto, auto),
    align: center,
    stroke: (x, y) => if x == 0 {(right: 0.75pt)},
    table.hline(),
    "OS", "Ubuntu 22.04.5 LTS",
    "CPU", "12th Gen Intel(R) Core(TM) i7-1280P",
    "Memory", "32 GB",
    "Compiler", "rustc 1.92.0-nightly (57ef8d642 2025-10-15)",
    table.hline(),
  )
]

また, 使用したソースコードは
https://github.com/kanarus/MathematicalEngineeringExperiment2025/tree/report2/chapter2
から参照できる.

本レポートでは, Rust 言語によって行列構造体およびアルゴリズムを実装しており,
誤差を調べるための参照実装には nalgebra 0.34.1 (https://docs.rs/nalgebra/0.34.1),
実験結果のプロットには plotters 0.3.7 (https://docs.rs/plotters/0.3.7) を用いている.

= 課題１

== 問題設定

連立１次方程式 $A bold(x) = bold(b)$ の消去法 (Gaussian elimination) による数値解法を実装し,
$A$ の次元を $n = 100, 200, 400, 800$ と変化させながら実行して誤差や所要時間を調べる.

== 実装

ソースコード (@intro[] 章参照) の src/bin/ex1.rs が対応するファイルであり,
chapter2 ディレクトリで #text(font: "Mint Mono")[cargo run \-\-bin ex1 \-\-release] を実行することで実験を再現できる.

特に, アルゴリズムおよび解法の実装は以下のようになっている:

#pagebreak()

#sourcecode[```rust
    fn do_gaussian_elimination<const N: usize>(ab: &mut Matrix<N, {N + 1}>) {
        for k in 0..(N - 1) {
            let (i, _pivot) = (k..N)
                .map(|i| (i, ab[(i, k)]))
                .filter(|(_, value)| value.abs() > EPSILON)
                .max_by(|(_, a), (_, b)| f64::partial_cmp(&a.abs(), &b.abs()).expect("found NaN or Inf"))
                .expect("Matrix is singular");
            
            if i != k {
                ab.swap_rows(i, k);
            }
            
            for i in (k + 1)..N {
                let factor = ab[(i, k)] / ab[(k, k)];
                for j in k..(N + 1) {
                    ab[(i, j)] -= factor * ab[(k, j)];
                }
            }
        }
    }
    
    fn solve_by_gaussian_elimination<const N: usize>(a: &Matrix<N, N>, b: &Vector<N>) -> Vector<N> where [(); N + 1]: {
        let mut augmented_coefficient_matrix = Matrix::concat(a, b);
        do_gaussian_elimination(&mut augmented_coefficient_matrix);    
        back_substitution(
            &Matrix::<N, N>::from_fn(|i, j| augmented_coefficient_matrix[(i, j)]),
            &Vector::<N>::from_fn(|i, _| augmented_coefficient_matrix[(i, N)]),
        )
    }
```]

#text(font: "Mint Mono")[do_gaussian_elimination] 関数が $N times (N+1)$ 行列に対する消去法,
#text(font: "Mint Mono")[solve_by_gaussian_elimination] 関数が拡大係数行列に対してそれを用いる１次方程式の解法をそれぞれ実装している.

前者は概ね資料通りのアルゴリズムだが,

- 倍精度浮動小数点数を用いている点
- 0-based indexing を採用している点
- 非零判定に #text(font: "Mint Mono")[EPSILON] (ここでは $10^(-10)$) との比較を用いている点

が異なる.

== 実験結果

まず, 相対誤差は以下のように推移した:

#figure(
  image("plot/ex1/n100-relative_error.svg", width: 80%),
)

= 課題２, ３

= 課題４, ５

#v(2em)
#bibliography("report.bib")
