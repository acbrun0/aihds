use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_svm::Svm;
use std::fs;
use std::path::Path;

pub fn train(train: &Dataset<f64, bool>) -> Result<Svm<f64, bool>, linfa_svm::SvmError> {
    Svm::<f64, bool>::params().fit(train)
}

pub fn load(path: &Path) -> Result<Svm<f64, bool>, std::io::Error> {
    let bin = fs::read(path)?;
    let model: Svm<f64, bool> = bincode::deserialize(&bin).unwrap();
    Ok(model)
}
