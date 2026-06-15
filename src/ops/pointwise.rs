use crate::{Tensor, TensorError};

pub fn neg(a: &Tensor) -> Result<Tensor, TensorError> {
    mul(a, &Tensor::from_vec(vec![-1.0; a.numel()], a.shape())?)
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;
    Tensor::from_vec(
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a + b)
            .collect(),
        a.shape(),
    )
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    add(a, &neg(b)?)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;
    Tensor::from_vec(
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a * b)
            .collect(),
        a.shape(),
    )
}

pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;
    Tensor::from_vec(
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a.powf(*b))
            .collect(),
        a.shape(),
    )
}

pub fn min(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;
    Tensor::from_vec(
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a.min(*b))
            .collect(),
        a.shape(),
    )
}

pub fn max(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;
    Tensor::from_vec(
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a.max(*b))
            .collect(),
        a.shape(),
    )
}

pub fn abs(a: &Tensor) -> Result<Tensor, TensorError> {
    Tensor::from_vec(a.data().iter().map(|a| a.abs()).collect(), a.shape())
}

fn check_shapes(a: &Tensor, b: &Tensor) -> Result<(), TensorError> {
    if a.ndim() != b.ndim() {
        return Err(TensorError::RankMismatch {
            expected: a.ndim(),
            got: b.ndim(),
        });
    }
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2x2(data: &[f32]) -> Tensor {
        Tensor::from_vec(data.to_vec(), &[2, 2]).unwrap()
    }

    #[test]
    fn add_elementwise() {
        let a = t2x2(&[1., 2., 3., 4.]);
        let b = t2x2(&[5., 6., 7., 8.]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data(), &[6., 8., 10., 12.]);
    }

    #[test]
    fn sub_elementwise() {
        let a = t2x2(&[5., 6., 7., 8.]);
        let b = t2x2(&[1., 2., 3., 4.]);
        let c = sub(&a, &b).unwrap();
        assert_eq!(c.data(), &[4., 4., 4., 4.]);
    }
    #[test]
    fn mul_elementwise() {
        let a = t2x2(&[1., 2., 3., 4.]);
        let b = t2x2(&[2., 2., 2., 2.]);
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.data(), &[2., 4., 6., 8.]);
    }
    #[test]
    fn neg_unary() {
        let a = t2x2(&[1., -2., 3., -4.]);
        let c = neg(&a).unwrap();
        assert_eq!(c.data(), &[-1., 2., -3., 4.]);
    }
    #[test]
    fn abs_unary() {
        let a = t2x2(&[1., -2., 3., -4.]);
        let c = abs(&a).unwrap();
        assert_eq!(c.data(), &[1., 2., 3., 4.]);
    }
    #[test]
    fn pow_elementwise() {
        let a = t2x2(&[2., 3., 4., 5.]);
        let b = t2x2(&[2., 1., 0.5, 1.]);
        let c = pow(&a, &b).unwrap();
        assert_eq!(c.data(), &[4., 3., 2., 5.]);
    }
    #[test]
    fn min_elementwise() {
        let a = t2x2(&[1., 5., 3., 2.]);
        let b = t2x2(&[2., 3., -3., 8.]);
        let c = min(&a, &b).unwrap();
        assert_eq!(c.data(), &[1., 3., -3., 2.]);
    }
    #[test]
    fn max_elementwise() {
        let a = t2x2(&[1., 5., -3., 2.]);
        let b = t2x2(&[2., 3., 3., 8.]);
        let c = max(&a, &b).unwrap();
        assert_eq!(c.data(), &[2., 5., 3., 8.]);
    }

    #[test]
    fn binary_op_rank_mismatch() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3, 1]).unwrap(); // rank 3
        assert_eq!(
            mul(&a, &b).unwrap_err(),
            TensorError::RankMismatch {
                expected: 2,
                got: 3
            }
        );
    }
    #[test]
    fn binary_op_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.; 4], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.; 6], &[2, 3]).unwrap();
        assert_eq!(
            add(&a, &b).unwrap_err(),
            TensorError::ShapeMismatch {
                expected: vec![2, 2],
                got: vec![2, 3],
            }
        );
    }
    #[test]
    fn preserves_shape() {
        let a = t2x2(&[1., 2., 3., 4.]);
        let b = t2x2(&[1., 1., 1., 1.]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
}
