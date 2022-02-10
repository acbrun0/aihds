use std::fs;
use std::path::Path;

pub mod svm {
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
}

pub mod trees {
    use linfa::dataset::Dataset;
    use linfa::prelude::*;
    use linfa_trees::{DecisionTree, SplitQuality};
    use std::fs;
    use std::path::Path;

    pub fn train(
        train: &Dataset<f64, bool>,
    ) -> std::result::Result<DecisionTree<f64, bool>, Error> {
        DecisionTree::params()
            .split_quality(SplitQuality::Gini)
            .max_depth(Some(100))
            .min_weight_split(1.0)
            .min_weight_leaf(1.0)
            .fit(train)
    }

    pub fn load(path: &Path) -> Result<DecisionTree<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        let model: DecisionTree<f64, bool> = bincode::deserialize(&bin).unwrap();
        Ok(model)
    }
}

// pub mod logistic {
//     use linfa::dataset::Dataset;
//     use linfa::prelude::*;
//     use linfa_logistic::FittedLogisticRegression;
//     use linfa_logistic::LogisticRegression;

//     pub fn train(
//         train: &Dataset<f64, bool>,
//     ) -> Result<FittedLogisticRegression<f64, bool>, linfa_logistic::error::Error> {
//         LogisticRegression::default().max_iterations(150).fit(train)
//     }
// }

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
