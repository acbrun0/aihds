use std::fs;
use std::path::Path;

pub mod clustering {
    use crate::dataset;
    use linfa::traits::{Fit, Predict};
    use linfa_clustering::KMeans;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(
        paths: Vec<&Path>,
    ) -> Result<KMeans<f64, linfa_nn::distance::L2Dist>, linfa_clustering::KMeansError> {
        let train = match dataset::load(paths, None, true) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded train dataset with shape: {:?}",
            train.records.shape()
        );
        println!("Training model...");
        KMeans::params(2).fit(&train)
    }

    pub fn test(
        paths: Vec<&Path>,
        model: KMeans<f64, linfa_nn::distance::L2Dist>,
        scaler: Option<Vec<(f64, f64)>>,
    ) -> ((u32, u32, u32, u32), f64) {
        let start = Instant::now();
        let test = match dataset::load(paths, scaler, true) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded validation dataset with shape: {:?}",
            test.records.shape()
        );
        // Test model
        println!("Testing model...");
        let start = Instant::now();
        let pred = model.predict(&test);
        let mut tp: u32 = 0;
        let mut fp: u32 = 0;
        let mut tn: u32 = 0;
        let mut fal_n: u32 = 0;

        for (p, r) in pred.iter().zip(test.targets.into_raw_vec()) {
            if r {
                if *p == 1 {
                    tp += 1;
                } else {
                    fal_n += 1;
                }
            } else if *p == 0 {
                tn += 1;
            } else {
                fp += 1;
            }
        }

        (
            (tp, fp, tn, fal_n),
            test.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }
}

pub mod svm {
    use crate::dataset;
    use linfa::prelude::*;
    use linfa_preprocessing::linear_scaling::LinearScaler;
    use linfa_svm::Svm;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(
        paths: Vec<&Path>,
    ) -> (
        Result<Svm<f64, bool>, linfa_svm::SvmError>,
        LinearScaler<f64>,
    ) {
        // Load datasets
        let train = match dataset::load_unsupervised(paths, None, false) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded train dataset with shape: {:?}",
            train.records.shape()
        );
        let scaler = LinearScaler::standard().fit(&train).unwrap();
        let train = scaler.transform(train);
        println!("Training model...");
        (Svm::<f64, _>::params().fit(&train), scaler)
    }

    pub fn grid_search(
        train: &Dataset<f64, bool>,
        test: &Dataset<f64, bool>,
    ) -> Result<Vec<(String, f64, f64, f64)>, Error> {
        struct Parameters {
            c: [f64; 4],
            eps: [f64; 4],
            kernel: [String; 2],
        }
        let params = Parameters {
            c: [0.1, 1.0, 10.0, 100.0],
            eps: [0.00001, 0.000001, 0.0000001, 0.00000001],
            kernel: [String::from("linear"), String::from("gaussian")],
        };
        let mut results = Vec::<(String, f64, f64, f64)>::new();
        for kernel in params.kernel {
            if kernel == "gaussian" {
                for eps in params.eps {
                    for c in params.c {
                        match Svm::<f64, bool>::params()
                            .pos_neg_weights(c * 10.0, c)
                            .gaussian_kernel(eps)
                            .fit(train)
                        {
                            Ok(model) => match model.predict(test).confusion_matrix(test) {
                                Ok(confusion_matrix) => {
                                    results.push((
                                        kernel.clone(),
                                        eps,
                                        c,
                                        confusion_matrix.f1_score().into(),
                                    ));
                                }
                                Err(why) => panic!("Could not compute confusion matrix: {}", why),
                            },
                            Err(why) => panic!(
                                "Could not train SVM with parameters {}, {}, and {}: {}",
                                kernel, eps, c, why
                            ),
                        }
                    }
                }
            } else if kernel == "linear" {
                for eps in params.eps {
                    for c in params.c {
                        match Svm::<f64, bool>::params()
                            .pos_neg_weights(c * 10.0, c)
                            .fit(train)
                        {
                            Ok(model) => match model.predict(test).confusion_matrix(test) {
                                Ok(confusion_matrix) => {
                                    results.push((
                                        kernel.clone(),
                                        eps,
                                        c,
                                        confusion_matrix.f1_score().into(),
                                    ));
                                }
                                Err(why) => panic!("Could not compute confusion matrix: {}", why),
                            },
                            Err(why) => panic!(
                                "Could not train SVM with parameters {}, {}, and {}: {}",
                                kernel, eps, c, why
                            ),
                        }
                    }
                }
            }
        }
        results.sort_by(|x, y| y.3.partial_cmp(&x.3).unwrap());
        Ok(results)
    }

    pub fn test(
        paths: Vec<&Path>,
        model: Svm<f64, bool>,
        scaler: LinearScaler<f64>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let start = Instant::now();
        let test = match dataset::load(paths, None, false) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded validation dataset with shape: {:?}",
            test.records.shape()
        );
        let test = scaler.transform(test);
        // Test model
        println!("Testing model...");
        let pred = model.predict(&test);

        (
            pred.confusion_matrix(&test),
            test.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }

    pub fn load(path: &Path) -> Result<Svm<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        let model: Svm<f64, bool> = bincode::deserialize(&bin).unwrap();
        Ok(model)
    }
}

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
