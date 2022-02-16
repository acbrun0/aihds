mod dataset;
mod model;

use clap::Parser;
use linfa::prelude::*;
use model::{bayes, svm, trees};
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
    /// Type of model to train (supported: SVM, trees)
    #[clap(short, long)]
    model: Option<String>,
    /// Extracts features to CSV files
    #[clap(short, long)]
    extract_features: Option<String>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    // Write extracted features to a CSV file
    if args.extract_features.is_some() {
        let paths = args.extract_features.unwrap();
        let paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
        let paths: Vec<&Path> = paths.iter().map(Path::new).collect();
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
            match dataset::load(vec![path]) {
                Ok(data) => {
                    let fp = &format!(
                        "{}/{}.csv",
                        dir,
                        path.file_stem().unwrap().to_str().unwrap()
                    );
                    match dataset::write_features(Path::new(fp), &data) {
                        Ok(_) => (),
                        Err(why) => println!("Could not write features to {}: {}", fp, why),
                    }
                }
                Err(why) => println!("Could not load data from {}: {}", path.display(), why),
            }
        }
        return Ok(());
    }

    // Check if provided model type is test
    if args.load.is_none() {
        if args.model.is_none() {
            panic!("Did not provide a type of model to train.");
        } else {
            match args.train {
                Some(paths) => {
                    let train_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
                    let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
                    let modeltype = args.model.unwrap();
                    match modeltype.to_lowercase().as_str() {
                        "svm" => match svm::train(train_paths) {
                            Ok(model) => {
                                match model::save(&model, Path::new("models/svm")) {
                                    Ok(_) => (),
                                    Err(why) => println!("Could not save model: {}", why),
                                }
                                if let Some(paths) = args.test {
                                    let test_paths: Vec<&str> =
                                        paths.split(',').collect::<Vec<&str>>();
                                    let test_paths: Vec<&Path> =
                                        test_paths.iter().map(Path::new).collect();
                                    let (confusion_matrix, speed) = svm::test(test_paths, model);
                                    println!("Classification speed: {:.2} packets/s", speed);
                                    match confusion_matrix {
                                        Ok(cm) => {
                                            println!("{:#?}", cm);
                                            println!(
                                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                                        cm.accuracy() * 100.0,
                                                        cm.precision() * 100.0,
                                                        cm.recall() * 100.0,
                                                        cm.f1_score() * 100.0
                                                    );
                                        }
                                        Err(why) => {
                                            println!("Could not compute confusion matrix: {}", why)
                                        }
                                    }
                                }
                            }
                            Err(why) => panic!("Could not train model: {}", why),
                        },
                        "trees" => match trees::train(train_paths) {
                            Ok(model) => {
                                match model::save(&model, Path::new("models/trees")) {
                                    Ok(_) => (),
                                    Err(why) => println!("Could not save model: {}", why),
                                }
                                if let Some(paths) = args.test {
                                    let test_paths: Vec<&str> =
                                        paths.split(',').collect::<Vec<&str>>();
                                    let test_paths: Vec<&Path> =
                                        test_paths.iter().map(Path::new).collect();
                                    let (confusion_matrix, speed) = trees::test(test_paths, model);
                                    println!("Classification speed: {:.2} packets/s", speed);
                                    match confusion_matrix {
                                        Ok(cm) => {
                                            println!("{:#?}", cm);
                                            println!(
                                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                                        cm.accuracy() * 100.0,
                                                        cm.precision() * 100.0,
                                                        cm.recall() * 100.0,
                                                        cm.f1_score() * 100.0
                                                    );
                                        }
                                        Err(why) => {
                                            println!("Could not compute confusion matrix: {}", why)
                                        }
                                    }
                                }
                            }
                            Err(why) => panic!("Could not train model: {}", why),
                        },
                        "bayes" => match bayes::train(train_paths) {
                            Ok(model) => {
                                if let Some(paths) = args.test {
                                    let test_paths: Vec<&str> =
                                        paths.split(',').collect::<Vec<&str>>();
                                    let test_paths: Vec<&Path> =
                                        test_paths.iter().map(Path::new).collect();
                                    let (confusion_matrix, speed) = bayes::test(test_paths, model);
                                    println!("Classification speed: {:.2} packets/s", speed);
                                    match confusion_matrix {
                                        Ok(cm) => {
                                            println!("{:#?}", cm);
                                            println!(
                                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                                        cm.accuracy() * 100.0,
                                                        cm.precision() * 100.0,
                                                        cm.recall() * 100.0,
                                                        cm.f1_score() * 100.0
                                                    );
                                        }
                                        Err(why) => {
                                            println!("Could not compute confusion matrix: {}", why)
                                        }
                                    }
                                }
                            }
                            Err(why) => panic!("Could not train model: {}", why),
                        },
                        _ => panic!("Unsupported model type: {}", modeltype),
                    }
                }
                None => panic!("Did not provide path to train datasets."),
            }
        };
    } else {
        // Load model
        println!("Loading model...");
        let modelpath = args.load.unwrap();
        if modelpath.contains("svm") {
            match svm::load(Path::new(&modelpath)) {
                Ok(model) => {
                    if let Some(paths) = args.test {
                        let test_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
                        let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                        let (confusion_matrix, speed) = svm::test(test_paths, model);
                        println!("Classification speed: {:.2} packets/s", speed);
                        match confusion_matrix {
                            Ok(cm) => {
                                println!("{:#?}", cm);
                                println!(
                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                        cm.accuracy() * 100.0,
                                        cm.precision() * 100.0,
                                        cm.recall() * 100.0,
                                        cm.f1_score() * 100.0
                                    );
                            }
                            Err(why) => {
                                println!("Could not compute confusion matrix: {}", why)
                            }
                        }
                    }
                }
                Err(why) => panic!("Could not load model: {}", why),
            };
        } else if modelpath.contains("trees") {
            match trees::load(Path::new(&modelpath)) {
                Ok(model) => {
                    if let Some(paths) = args.test {
                        let test_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
                        let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                        let (confusion_matrix, speed) = trees::test(test_paths, model);
                        println!("Classification speed: {:.2} packets/s", speed);
                        match confusion_matrix {
                            Ok(cm) => {
                                println!("{:#?}", cm);
                                println!(
                                        "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF1-Score: {:.3}%",
                                        cm.accuracy() * 100.0,
                                        cm.precision() * 100.0,
                                        cm.recall() * 100.0,
                                        cm.f1_score() * 100.0
                                    );
                            }
                            Err(why) => {
                                println!("Could not compute confusion matrix: {}", why)
                            }
                        }
                    }
                }
                Err(why) => panic!("Could not load model: {}", why),
            };
        } else {
            panic!("Could not load model from {}", modelpath)
        }
    }
    Ok(())
}
