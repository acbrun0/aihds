use std::fs;
use std::path::Path;

pub mod svm {
    use linfa::prelude::*;
    use linfa_svm::Svm;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;
    use ndarray::{ArrayBase, ViewRepr, Dim};

    pub fn train(dataset: &Dataset<f64, ()>) -> Result<Svm<f64, bool>, linfa_svm::SvmError> {
        Svm::<f64, _>::params()
            // .gaussian_kernel(1.0)
            .polynomial_kernel(0.0, 7.0)
            .nu_weight(0.001)
            .fit(dataset)
    }

    pub fn predict(model: &Svm<f64, bool>, features: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>) -> bool{
        model.predict(features)
    }

    pub fn test(
        dataset: &Dataset<f64, bool>,
        model: Svm<f64, bool>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let start = Instant::now();
        let pred = model.predict(dataset);
        (
            pred.confusion_matrix(dataset),
            // pred.mapv(|p| !p).confusion_matrix(dataset),
            dataset.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }

    pub fn load(path: &Path) -> Result<Svm<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        let model: Svm<f64, bool> = bincode::deserialize(&bin).unwrap();
        Ok(model)
    }

    #[allow(clippy::type_complexity)]
    pub fn grid_search(
        train: &Dataset<f64, ()>,
        test: &Dataset<f64, bool>,
    ) -> Result<Vec<(String, f64, f64, f64, f64, u8, u8, f32)>, Error> {
        struct Parameters {
            c: [f64; 4],
            eps: [f64; 5],
            nu: [f64; 10],
            degree: [u8; 4],
            cnst: [u8; 5],
            kernel: [String; 3],
        }
        let params = Parameters {
            c: [0.1, 1.0, 5.0, 10.0],
            eps: [0.0001, 0.01, 0.1, 1.0, 10.0],
            nu: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            degree: [2, 3, 4, 5],
            cnst: [1, 2, 3, 4, 5],
            kernel: [
                String::from("linear"),
                String::from("poly"),
                String::from("gaussian"),
            ],
        };
        let mut results = Vec::<(String, f64, f64, f64, f64, u8, u8, f32)>::new();
        for kernel in params.kernel {
            if kernel == "gaussian" {
                for eps in params.eps {
                    for c1 in params.c {
                        for c2 in params.c {
                            for nu in params.nu {
                                match Svm::<f64, _>::params().pos_neg_weights(c1, c2).gaussian_kernel(eps).nu_weight(nu).fit(train)
                                {
                                    Ok(model) => match model.predict(test).confusion_matrix(test) {
                                        Ok(confusion_matrix) => {
                                            println!(
                                                "Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {:.3}; {:.3}",
                                                kernel,
                                                eps,
                                                c1,
                                                c2,
                                                nu,
                                                confusion_matrix.f1_score() * 100.0,
                                                confusion_matrix.accuracy() * 100.0
                                            );
                                            results.push((
                                                kernel.clone(),
                                                eps,
                                                c1,
                                                c2,
                                                nu,
                                                0,
                                                0,
                                                confusion_matrix.f1_score(),
                                            ));
                                        }
                                        Err(why) => {
                                            panic!("Could not compute confusion matrix: {}", why)
                                        }
                                    },
                                    Err(why) => panic!("Could not train SVM with parameters - Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {}", kernel, eps, c1, c2, nu, why),
                                }
                            }
                        }
                    }
                }
            } else if kernel == "linear" {
                for nu in params.nu {
                    for c1 in params.c {
                        for c2 in params.c {
                            match Svm::<f64, _>::params()
                                .pos_neg_weights(c1, c2)
                                .nu_weight(nu)
                                .fit(train)
                            {
                                Ok(model) => match model.predict(test).confusion_matrix(test) {
                                    Ok(confusion_matrix) => {
                                        println!(
                                            "Nu: {}; C-pos: {}; C-neg: {} - {:.3}; {:.3}",
                                            nu,
                                            c1,
                                            c2,
                                            confusion_matrix.f1_score() * 100.0,
                                            confusion_matrix.accuracy() * 100.0
                                        );
                                        if !(confusion_matrix.f1_score().is_nan()
                                            || confusion_matrix.accuracy().is_nan())
                                        {
                                            results.push((
                                                kernel.clone(),
                                                -1.0,
                                                nu,
                                                c1,
                                                c2,
                                                0,
                                                0,
                                                confusion_matrix.f1_score(),
                                            ));
                                        }
                                    }
                                    Err(why) => {
                                        panic!("Could not compute confusion matrix: {}", why)
                                    }
                                },
                                Err(why) => panic!(
                                    "Could not train SVM with parameters {}, {}, and {}: {}",
                                    nu, c1, c2, why
                                ),
                            }
                        }
                    }
                }
            } else if kernel == "poly" {
                for nu in params.nu {
                    for c1 in params.c {
                        for c2 in params.c {
                            for c in params.cnst {
                                for d in params.degree {
                                    match Svm::<f64, _>::params().polynomial_kernel(c as f64, d as f64)
                                        .pos_neg_weights(c1, c2)
                                        .nu_weight(nu)
                                        .fit(train)
                                    {
                                        Ok(model) => match model.predict(test).confusion_matrix(test) {
                                            Ok(confusion_matrix) => {
                                                println!(
                                                    "Nu: {}; C-pos: {}; C-neg: {}; Constant: {}; Degree: {} - {:.3}; {:.3}",
                                                    nu,
                                                    c1,
                                                    c2,
                                                    c,
                                                    d,
                                                    confusion_matrix.f1_score() * 100.0,
                                                    confusion_matrix.accuracy() * 100.0
                                                );
                                                if !(confusion_matrix.f1_score().is_nan()
                                                    || confusion_matrix.accuracy().is_nan())
                                                {
                                                    results.push((
                                                        kernel.clone(),
                                                        -1.0,
                                                        nu,
                                                        c1,
                                                        c2,
                                                        c,
                                                        d,
                                                        confusion_matrix.f1_score(),
                                                    ));
                                                }
                                            }
                                            Err(why) => {
                                                panic!("Could not compute confusion matrix: {}", why)
                                            }
                                        },
                                        Err(why) => panic!(
                                            "Could not train SVM with parameters {}, {}, and {}: {}",
                                            nu, c1, c2, why
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        results.sort_by(|x, y| y.7.partial_cmp(&x.7).unwrap());
        Ok(results)
    }
}

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
