use crate::{Tensor, TensorError};

pub fn sum(a: &Tensor) -> Result<Tensor, TensorError> {
    Tensor::from_vec(vec![a.data().iter().sum::<f32>()], &[1])
}

pub fn mean(a: &Tensor) -> Result<Tensor, TensorError> {
    Tensor::from_vec(vec![sum(a)?.data()[0] / a.numel() as f32], &[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_all_elements() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let s = sum(&t).unwrap();
        assert_eq!(s.data(), &[10.]);
        assert_eq!(s.shape(), &[1]);
    }

    #[test]
    fn mean_all_elements() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        let m = mean(&t).unwrap();
        assert_eq!(m.data(), &[2.5]);
    }
}
