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
        let train = match dataset::load(paths) {
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
    ) -> ((u32, u32, u32, u32), f64) {
        let start = Instant::now();
        let test = match dataset::load(paths) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded validation dataset with shape: {:?}",
            test.records.shape()
        );
        // Test model
        println!("Testing model...");
        let pred = model.predict(&test);
        let end = start.elapsed().as_secs_f64();
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

        ((tp, fp, tn, fal_n), test.records.shape()[0] as f64 / end)
    }
}

pub mod bayes {
    use crate::dataset;
    use linfa::prelude::*;
    use linfa_bayes::GaussianNb;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(
        paths: Vec<&Path>,
    ) -> Result<linfa_bayes::GaussianNb<f64, bool>, linfa_bayes::NaiveBayesError> {
        // Load datasets
        let train = match dataset::load(paths) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded train dataset with shape: {:?}",
            train.records.shape()
        );
        println!("Training model...");
        GaussianNb::params().fit(&train)
    }

    pub fn test(
        paths: Vec<&Path>,
        model: GaussianNb<f64, bool>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let test = match dataset::load(paths) {
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

        (
            pred.confusion_matrix(&test),
            test.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }
}

pub mod svm {
    use crate::dataset;
    use linfa::prelude::*;
    use linfa_svm::Svm;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(paths: Vec<&Path>) -> Result<Svm<f64, bool>, linfa_svm::SvmError> {
        // Load datasets
        let train = match dataset::load(paths) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded train dataset with shape: {:?}",
            train.records.shape()
        );
        println!("Training model...");
        Svm::<f64, bool>::params().fit(&train)
    }

    pub fn test(
        paths: Vec<&Path>,
        model: Svm<f64, bool>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let test = match dataset::load(paths) {
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

pub mod trees {
    use crate::dataset;
    use linfa::prelude::*;
    use linfa_trees::{DecisionTree, SplitQuality};
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(paths: Vec<&Path>) -> std::result::Result<DecisionTree<f64, bool>, Error> {
        // Load datasets
        let train = match dataset::load(paths) {
            Ok(dataset) => dataset,
            Err(why) => panic!("Could not read file: {}", why),
        };
        println!(
            "Loaded train dataset with shape: {:?}",
            train.records.shape()
        );
        println!("Training model...");
        DecisionTree::params()
            .split_quality(SplitQuality::Gini)
            .max_depth(Some(100))
            .min_weight_split(1.0)
            .min_weight_leaf(1.0)
            .fit(&train)
    }

    pub fn test(
        paths: Vec<&Path>,
        model: DecisionTree<f64, bool>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let test = match dataset::load(paths) {
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

        let mut tikz = File::create("decision_tree.tex").unwrap();
        tikz.write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
            .unwrap();

        (
            pred.confusion_matrix(&test),
            test.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }

    pub fn load(path: &Path) -> Result<DecisionTree<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        let model: DecisionTree<f64, bool> = bincode::deserialize(&bin).unwrap();
        Ok(model)
    }
}

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
