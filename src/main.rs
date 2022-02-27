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
    #[clap(short, long)]
    train: Option<String>,
    /// Paths to the datasets required for testing the model, separated by ','
    #[clap(short, long)]
    test: Option<String>,
    /// Path to model to be loaded
    #[clap(short, long)]
    load: Option<String>,
    /// Extracts features to CSV files
    #[clap(short, long)]
    extract_features: Option<String>,
    /// Perform grid search optimization on SVM
    #[clap(short, long)]
    grid_search: bool,
    /// Use libsvm format
    #[clap(short, long)]
    libsvm: bool,
}

#[allow(dead_code)]
const ALL_TRAIN_PATHS: [&str; 11] = [
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_D_0.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_D_1.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_D_2.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_S_0.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_S_1.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\0_Training\\Pre_train_S_2.csv",
    ".\\datasets\\car_hacking\\DoS\\train.csv",
    ".\\datasets\\car_hacking\\fuzzy\\train.csv",
    ".\\datasets\\car_hacking\\gear\\train.csv",
    ".\\datasets\\car_hacking\\rpm\\train.csv",
    ".\\datasets\\car_hacking\\normal\\dataset.csv ",
];

#[allow(dead_code)]
const ALL_TEST_PATHS: [&str; 7] = [
    ".\\datasets\\ieee_challenge\\0_Preliminary\\1_Submission\\Pre_submit_D.csv",
    ".\\datasets\\ieee_challenge\\0_Preliminary\\1_Submission\\Pre_submit_S.csv",
    ".\\datasets\\ieee_challenge\\1_Final\\Fin_host_session_submit_S.csv",
    ".\\datasets\\car_hacking\\DoS\\test.csv",
    ".\\datasets\\car_hacking\\rpm\\test.csv",
    ".\\datasets\\car_hacking\\gear\\test.csv",
    ".\\datasets\\car_hacking\\fuzzy\\test.csv",
];

fn main() -> Result<(), Error> {
    let args = Args::parse();

    // Write extracted features to file
    if let Some(paths) = args.extract_features {
        let paths = paths.split(',').collect::<Vec<&str>>();
        let paths = paths.iter().map(Path::new).collect();
        if args.libsvm {
            let scaler = match fs::read("models/scaler") {
                Ok(scaler) => match bincode::deserialize(&scaler) {
                    Ok(scaler) => Some(scaler),
                    Err(why) => panic!("Could not deserialize scaler: {}", why)
                }
                Err(_) => None
            };
            match dataset::load_unsupervised(paths, scaler) {
                Ok((mut dataset, _)) => match dataset::write_features(
                    Path::new("libsvm/features.txt"),
                    &mut dataset,
                    true,
                ) {
                    Ok(_) => (),
                    Err(why) => println!("Could not write features: {}", why),
                },
                Err(why) => panic!("Could not load dataset: {}", why),
            };
        } else {
            for path in paths {
                let dir = &format!(
                    "features/{}",
                    path.parent()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                );
                match std::fs::create_dir_all(Path::new(dir)) {
                    Ok(_) => (),
                    Err(why) => panic!("Could not create features directory: {}", why),
                };
                match dataset::load_unsupervised(vec![path], None) {
                    Ok((mut dataset, _)) => {
                        let fp = &format!(
                            "{}/{}.csv",
                            dir,
                            path.file_stem().unwrap().to_str().unwrap()
                        );
                        match dataset::write_features(Path::new(fp), &mut dataset, false) {
                            Ok(_) => (),
                            Err(why) => println!("Could not write features to {}: {}", fp, why),
                        }
                    }
                    Err(why) => panic!("Could not load dataset: {}", why),
                };
            }
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
