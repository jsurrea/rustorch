use crate::{Tensor, TensorError};

/// Matrix multiplication
///
/// For rank >= 2, batch dimensions (`shape[..-2]`) must match exactly.
/// The last two dimensions are treated as `(M, K) @ (K, N) -> (M, N)`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(matmul_shape_mismatch(a_shape, b_shape));
    }

    if a.ndim() != b.ndim() {
        return Err(matmul_shape_mismatch(a_shape, b_shape));
    }

    if a_shape[a.ndim() - 1] != b_shape[b.ndim() - 2] {
        return Err(matmul_shape_mismatch(a_shape, b_shape));
    }

    let batch = &a_shape[..a.ndim() - 2];
    if batch != &b_shape[..b.ndim() - 2] {
        return Err(matmul_shape_mismatch(a_shape, b_shape));
    }

    let m = a_shape[a.ndim() - 2];
    let k = a_shape[a.ndim() - 1];
    let n = b_shape[b.ndim() - 1];

    let batch_count: usize = batch.iter().product();
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;

    let mut out_data = vec![0.0; batch_count * c_batch_stride];
    let a_data = a.data();
    let b_data = b.data();

    for batch_idx in 0..batch_count {
        let a_off = batch_idx * a_batch_stride;
        let b_off = batch_idx * b_batch_stride;
        let c_off = batch_idx * c_batch_stride;

        matmul_2d(
            &a_data[a_off..a_off + a_batch_stride],
            &b_data[b_off..b_off + b_batch_stride],
            &mut out_data[c_off..c_off + c_batch_stride],
            m,
            k,
            n,
        );
    }

    let mut out_shape = batch.to_vec();
    out_shape.extend([m, n]);
    Tensor::from_vec(out_data, &out_shape)
}

fn matmul_shape_mismatch(a_shape: &[usize], b_shape: &[usize]) -> TensorError {
    TensorError::MatmulShapeMismatch {
        a_shape: a_shape.to_vec(),
        b_shape: b_shape.to_vec(),
    }
}

fn matmul_2d(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matmul_2x2() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5., 6., 7., 8.], &[2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        // [[19, 22], [43, 50]]
        assert_eq!(c.data(), &[19., 22., 43., 50.]);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn matmul_non_square() {
        // (2,3) @ (3,2) -> (2,2)
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[22., 28., 49., 64.]);
    }

    #[test]
    fn matmul_inner_dim_mismatch() {
        let a = Tensor::from_vec(vec![1.; 6], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.; 8], &[2, 4]).unwrap();
        assert_eq!(
            matmul(&a, &b).unwrap_err(),
            TensorError::MatmulShapeMismatch {
                a_shape: vec![2, 3],
                b_shape: vec![2, 4],
            }
        );
    }

    #[test]
    fn matmul_batched_3d() {
        // batch 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
        // batch 1: [[1,0],[0,1]] @ [[2,3],[4,5]] = [[2,3],[4,5]]
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 1., 0., 0., 1.], &[2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1., 2., 3., 4., 2., 3., 4., 5.], &[2, 2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(c.data(), &[7., 10., 15., 22., 2., 3., 4., 5.]);
    }

    #[test]
    fn matmul_batch_mismatch() {
        let a = Tensor::from_vec(vec![1.; 24], &[2, 3, 4]).unwrap();
        let b = Tensor::from_vec(vec![1.; 60], &[3, 4, 5]).unwrap();
        assert_eq!(
            matmul(&a, &b).unwrap_err(),
            TensorError::MatmulShapeMismatch {
                a_shape: vec![2, 3, 4],
                b_shape: vec![3, 4, 5],
            }
        );
    }

    #[test]
    fn matmul_rank_mismatch() {
        let a = Tensor::from_vec(vec![1.; 24], &[2, 3, 4]).unwrap();
        let b = Tensor::from_vec(vec![1.; 20], &[4, 5]).unwrap();
        assert_eq!(
            matmul(&a, &b).unwrap_err(),
            TensorError::MatmulShapeMismatch {
                a_shape: vec![2, 3, 4],
                b_shape: vec![4, 5],
            }
        );
    }
}
