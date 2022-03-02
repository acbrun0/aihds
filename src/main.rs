mod dataset;
mod model;

use clap::Parser;
use linfa::prelude::*;
use model::svm;
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    /// Paths to the datasets required for training the model, separated by ','
    #[clap(long)]
    train: Option<String>,
    /// Paths to the datasets required for testing the model, separated by ','
    #[clap(long)]
    test: Option<String>,
    /// Path to model to be loaded
    #[clap(long)]
    load: Option<String>,
    /// Extracts features to CSV files
    #[clap(long)]
    extract_features: bool,
    /// Perform grid search optimization on SVM
    #[clap(long)]
    grid_search: bool,
    /// Use libsvm format
    #[clap(long)]
    libsvm: bool,
    /// Join features into a single file
    #[clap(long)]
    join: bool,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    if args.extract_features {
        match std::fs::create_dir_all(Path::new("features")) {
            Ok(_) => (),
            Err(why) => panic!("Could not create features directory: {}", why),
        };
        if let Some(paths) = args.train {
            let paths = paths.split(',').collect::<Vec<&str>>();
            let paths = paths.iter().map(Path::new).collect();
            if args.join {
                match dataset::load_unsupervised(paths, None) {
                    Ok((mut dataset, _)) => {
                        match dataset::write_features_unsupervised(
                            Path::new(&format!(
                                "{}/features.txt",
                                if args.libsvm { "libsvm" } else { "features" }
                            )),
                            &mut dataset,
                            args.libsvm,
                        ) {
                            Ok(_) => (),
                            Err(why) => println!("Could not write features: {}", why),
                        }
                    }
                    Err(why) => panic!("Could not load dataset: {}", why),
                }
            } else {
                for path in paths {
                    match dataset::load_unsupervised(vec![path], None) {
                        Ok((mut dataset, _)) => {
                            let fp = format!(
                                "features/{}.csv",
                                path.file_stem().unwrap().to_str().unwrap()
                            );
                            match dataset::write_features_unsupervised(Path::new(&fp), &mut dataset, false) {
                                Ok(_) => (),
                                Err(why) => {
                                    println!("Could not write features to {}: {}", fp, why)
                                }
                            }
                        }
                        Err(why) => panic!("Could not load dataset: {}", why),
                    };
                }
            };
        }
        if let Some(paths) = args.test {
            let scaler: Option<Vec<(f64, f64)>> = match fs::read("models/scaler") {
                Ok(scaler) => match bincode::deserialize(&scaler) {
                    Ok(scaler) => Some(scaler),
                    Err(why) => panic!("Could not deserialize scaler: {}", why),
                },
                Err(_) => {
                    println!("Scaler not found in models/scaler. Proceeding without.");
                    None
                }
            };
            let paths = paths.split(',').collect::<Vec<&str>>();
            let paths = paths.iter().map(Path::new).collect();
            if args.join {
                match dataset::load(paths, scaler) {
                    Ok((mut dataset, _)) => {
                        match dataset::write_features(
                            Path::new(&format!(
                                "{}/targets.txt",
                                if args.libsvm { "libsvm" } else { "features" }
                            )),
                            &mut dataset,
                            args.libsvm,
                        ) {
                            Ok(_) => (),
                            Err(why) => println!("Could not write features: {}", why),
                        }
                    }
                    Err(why) => panic!("Could not load dataset: {}", why),
                }
            } else {
                let mut scaler_copy = scaler;
                for path in paths {
                    match dataset::load(vec![path], scaler_copy) {
                        Ok((mut dataset, scaler)) => {
                            scaler_copy = Some(scaler);
                            let fp = format!(
                                "features/{}.csv",
                                path.file_stem().unwrap().to_str().unwrap()
                            );
                            match dataset::write_features(Path::new(&fp), &mut dataset, false) {
                                Ok(_) => (),
                                Err(why) => {
                                    println!("Could not write features to {}: {}", fp, why)
                                }
                            }
                        }
                        Err(why) => panic!("Could not load dataset: {}", why),
                    };
                }
            };
        }
        return Ok(());
    }

    if args.grid_search {
        match args.train {
            Some(paths) => {
                let train_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
                let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
                if let Some(paths) = args.test {
                    let test_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
                    let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                    match dataset::load_unsupervised(train_paths, None) {
                        Ok((train_dataset, scaler)) => {
                            match dataset::load(test_paths, Some(scaler)) {
                                Ok((test_dataset, _)) => {
                                    match svm::grid_search(&train_dataset, &test_dataset) {
                                        Ok(results) => println!("{:#?}", results),
                                        Err(why) => {
                                            panic!("Could not perform grid search: {}", why)
                                        }
                                    }
                                }
                                Err(why) => panic!("Could not load test datasets: {}", why),
                            }
                        }
                        Err(why) => panic!("Could not load train datasets: {}", why),
                    }
                }
            }
            None => panic!("Did not provide path to train datasets."),
        }
    } else if args.load.is_none() {
        if let Some(paths) = args.train {
            let train_paths: Vec<&str> = paths.split(',').collect();
            let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
            match dataset::load_unsupervised(train_paths, None) {
                Ok((train_dataset, scaler)) => match svm::train(&train_dataset) {
                    Ok(model) => {
                        match model::save(&model, Path::new("models/svm")) {
                            Ok(_) => (),
                            Err(why) => println!("Could not save model: {}", why),
                        }
                        if let Some(paths) = args.test {
                            let test_paths: Vec<&str> = paths.split(',').collect();
                            let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                            match dataset::load(test_paths, Some(scaler)) {
                                Ok((test_dataset, _)) => {
                                    let (result, speed) = svm::test(&test_dataset, model);
                                    println!("Classification speed: {:.2} packets/s", speed);
                                    match result {
                                        Ok(confusion_matrix) => {
                                            println!("{:#?}", confusion_matrix);
                                            println!(
                                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                                        confusion_matrix.accuracy() * 100.0,
                                                        confusion_matrix.precision() * 100.0,
                                                        confusion_matrix.recall() * 100.0,
                                                        confusion_matrix.f1_score() * 100.0
                                                    );
                                        }
                                        Err(why) => {
                                            panic!("Could not compute confusion matrix: {}", why)
                                        }
                                    }
                                }
                                Err(why) => panic!("Could not load test datasets: {}", why),
                            }
                        }
                    }
                    Err(why) => panic!("Could not train model: {}", why),
                },
                Err(why) => panic!("Could not load train datasets: {}", why),
            }
        } else {
            panic!("Did not provide path to train datasets.")
        }
    } else {
        // Load model
        println!("Loading model...");
        let modelpath = args.load.unwrap();
        match svm::load(Path::new(&modelpath)) {
            Ok(model) => {
                if let Some(paths) = args.test {
                    let test_paths: Vec<&str> = paths.split(',').collect();
                    let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                    let scaler = bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
                    match dataset::load(test_paths, Some(scaler)) {
                        Ok((test_dataset, _)) => {
                            let (result, speed) = svm::test(&test_dataset, model);
                            println!("Classification speed: {:.2} packets/s", speed);
                            match result {
                                Ok(confusion_matrix) => {
                                    println!("{:#?}", confusion_matrix);
                                    println!(
                                            "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                            confusion_matrix.accuracy() * 100.0,
                                            confusion_matrix.precision() * 100.0,
                                            confusion_matrix.recall() * 100.0,
                                            confusion_matrix.f1_score() * 100.0
                                        );
                                }
                                Err(why) => panic!("Could not compute confusion matrix: {}", why),
                            }
                        }
                        Err(why) => panic!("Could not load test datasets: {}", why),
                    }
                }
            }
            Err(why) => panic!("Could not load model: {}", why),
        };
    }
    Ok(())
}
