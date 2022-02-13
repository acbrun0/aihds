mod dataset;
mod model;

use clap::Parser;
use linfa::prelude::*;
// use model::logistic;
use model::{svm, trees};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    /// Paths to the datasets required for training the model, separated by ','
    #[clap(short, long)]
    datasets: String,
    /// Path to model to be loaded
    #[clap(short, long)]
    load: Option<String>,
    /// Type of model to train (supported: SVM, trees)
    #[clap(short, long)]
    model: Option<String>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let paths: Vec<&str> = args.datasets.split(',').collect();
    let paths: Vec<&Path> = paths.iter().map(Path::new).collect();
    let mut modeltype = String::new();

    // Check if provided model type is test
    if args.load.is_none() {
        if args.model.is_none() {
            panic!("Did not provide a type of model to train.");
        } else {
            modeltype = args.model.unwrap();
            match modeltype.to_lowercase().as_str() {
                "svm" => (),
                "trees" => (),
                _ => panic!("Unsupported model type: {}", modeltype),
            }
        }
    }

    // Load datasets
    let train_paths: Vec<PathBuf> = paths.iter().map(|p| p.join("train.csv")).collect();
    let train_paths: Vec<&Path> = train_paths.iter().map(|p| p.as_path()).collect();
    let train = match dataset::load::load(train_paths) {
        Ok(dataset) => dataset,
        Err(why) => panic!("Could not read file: {}", why),
    };
    println!(
        "Loaded train dataset with shape: {:?}",
        train.records.shape()
    );
    let test_paths: Vec<PathBuf> = paths.iter().map(|p| p.join("test.csv")).collect();
    let test_paths: Vec<&Path> = test_paths.iter().map(|p| p.as_path()).collect();
    let test = match dataset::load::load(test_paths) {
        Ok(dataset) => dataset,
        Err(why) => panic!("Could not read file: {}", why),
    };
    println!(
        "Loaded validation dataset with shape: {:?}",
        test.records.shape()
    );

    if args.load.is_none() {
        // Train model
        match modeltype.to_lowercase().as_str() {
            "svm" => {
                println!("Training model...");
                let model = match svm::train(&train) {
                    Ok(model) => model,
                    Err(why) => panic!("Error training model: {}", why),
                };
                // Test model
                println!("Testing model...");
                let start = Instant::now();
                let pred = model.predict(&test);
                let duration = start.elapsed();
                let confusion_matrix = pred.confusion_matrix(&test)?;
                println!("{:#?}", confusion_matrix);
                println!(
                    "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF-Score: {:.3}%",
                    confusion_matrix.accuracy() * 100.0,
                    confusion_matrix.precision() * 100.0,
                    confusion_matrix.recall() * 100.0,
                    confusion_matrix.f_score(0.5) * 100.0
                );
                println!(
                    "Classification speed: {:.2} packets/s",
                    test.records.shape()[0] as f64 / duration.as_secs_f64()
                );
            }
            "trees" => {
                println!("Training model...");
                let model = match trees::train(&train) {
                    Ok(model) => model,
                    Err(why) => panic!("Error training model: {}", why),
                };

                // Print model info
                let mut tikz = File::create("decision_tree_example.tex").unwrap();
                tikz.write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
                    .unwrap();
                // Test model
                println!("Testing model...");
                let start = Instant::now();
                let pred = model.predict(&test);
                let duration = start.elapsed();
                let confusion_matrix = pred.confusion_matrix(&test)?;
                println!("{:#?}", confusion_matrix);
                println!(
                    "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF-Score: {:.3}%",
                    confusion_matrix.accuracy() * 100.0,
                    confusion_matrix.precision() * 100.0,
                    confusion_matrix.recall() * 100.0,
                    confusion_matrix.f_score(0.5) * 100.0
                );
                println!(
                    "Classification speed: {:.2} packets/s",
                    test.records.shape()[0] as f64 / duration.as_secs_f64()
                );
                // Save model
                match model::save(&model, Path::new("models/trees")) {
                    Ok(_) => (),
                    Err(why) => println!("Could not save model: {}", why),
                }
            }
            // "logistic" => {
            //     println!("Training model...");
            //     let model = match logistic::train(&train) {
            //         Ok(model) => model,
            //         Err(why) => panic!("Error training model: {}", why),
            //     };
            //     // Test model
            //     println!("Testing model...");
            //     let pred = model.predict(&test);
            //     println!("Accuracy: {:?}", pred.confusion_matrix(&test)?.accuracy());
            // }
            _ => panic!("Unsupported model type: {}", modeltype),
        }
    } else {
        // Load model
        println!("Loading model...");
        let modelpath = args.load.unwrap();
        if modelpath.contains("svm") {
            let model = match svm::load(Path::new(&modelpath)) {
                Ok(m) => m,
                Err(why) => panic!("Could not load model: {}", why),
            };
            // Test model
            println!("Testing model...");
            let start = Instant::now();
            let pred = model.predict(&test);
            let duration = start.elapsed();
            let confusion_matrix = pred.confusion_matrix(&test)?;
            println!("{:#?}", confusion_matrix);
            println!(
                "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF-Score: {:.3}%",
                confusion_matrix.accuracy() * 100.0,
                confusion_matrix.precision() * 100.0,
                confusion_matrix.recall() * 100.0,
                confusion_matrix.f_score(0.5) * 100.0
            );
            println!(
                "Classification speed: {:.2} packets/s",
                test.records.shape()[0] as f64 / duration.as_secs_f64()
            );
        } else if modelpath.contains("trees") {
            let model = match trees::load(Path::new(&modelpath)) {
                Ok(m) => m,
                Err(why) => panic!("Could not load model: {}", why),
            };
            // Test model
            println!("Testing model...");
            let start = Instant::now();
            let pred = model.predict(&test);
            let duration = start.elapsed();
            let confusion_matrix = pred.confusion_matrix(&test)?;
            println!("{:#?}", confusion_matrix);
            println!(
                "Accuracy: {:.3}%\nPrecision: {:.3}%\nRecall: {:.3}%\nF-Score: {:.3}%",
                confusion_matrix.accuracy() * 100.0,
                confusion_matrix.precision() * 100.0,
                confusion_matrix.recall() * 100.0,
                confusion_matrix.f_score(0.5) * 100.0
            );
            println!(
                "Classification speed: {:.2} packets/s",
                test.records.shape()[0] as f64 / duration.as_secs_f64()
            );
        } else {
            panic!("Could not load model from {}", modelpath)
        }
    }

    Ok(())
}
