mod dataset;
mod model;

use clap::Parser;
use linfa::prelude::*;
use model::svm;
use std::path::Path;

#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    /// Paths to the datasets required for training the model, separated by ','
    #[clap(short, long)]
    datasets: String,
    /// Path to model to be loaded
    #[clap(short, long)]
    load: Option<String>,
    /// Type of model to train (supported: SVM)
    #[clap(short, long)]
    model: Option<String>,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    let paths: Vec<&str> = args.datasets.split(',').collect();
    let mut modeltype = String::new();

    // Check if provided model type is valid
    if args.load.is_none() {
        if args.model.is_none() {
            panic!("Did not provide a type of model to train.");
        } else {
            modeltype = args.model.unwrap();
            match modeltype.to_lowercase().as_str() {
                "svm" => (),
                _ => panic!("Unsupported model type: {}", modeltype),
            }
        }
    }

    // Load datasets
    let (train, valid) = match dataset::load::load(paths.iter().map(Path::new).collect()) {
        Ok(dataset) => dataset,
        Err(why) => panic!("Could not read file: {}", why),
    };
    println!(
        "Loaded train dataset with shape: {:?}",
        train.records.shape()
    );
    println!(
        "Loaded validation dataset with shape: {:?}",
        valid.records.shape()
    );

    // Load or train model
    let model;
    if args.load.is_none() {
        match modeltype.to_lowercase().as_str() {
            "svm" => {
                println!("Training model...");
                model = match svm::train(&train) {
                    Ok(model) => model,
                    Err(why) => panic!("Error training model: {}", why),
                };
                match model::save(&model, Path::new("models/svm")) {
                    Ok(_) => (),
                    Err(why) => println!("Could not save model: {}", why),
                }
            }
            _ => panic!("Unsupported model type: {}", modeltype),
        }
    } else {
        println!("Loading model...");
        model = match svm::load(Path::new(&args.load.unwrap())) {
            Ok(m) => m,
            Err(why) => panic!("Could not load model: {}", why),
        }
    }

    // Test model
    println!("Testing model...");
    let pred = model.predict(&valid);
    println!("Accuracy: {:?}", pred.confusion_matrix(&valid)?.accuracy());

    Ok(())
}
