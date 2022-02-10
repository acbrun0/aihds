use clap::Parser;
use csv::Error;
use csv::StringRecord;
use linfa::dataset::Dataset;
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use std::char;
use std::path::Path;

type TrainValSet = (Dataset<f64, bool>, Dataset<f64, bool>);

#[derive(Parser)]
#[clap(author, version, about)]
struct Args {
    /// Paths to the datasets required for training the model, separated by ','
    #[clap(short, long)]
    paths: String,
}

// struct Packet {
//     timestamp: f64,
//     id: f64,
//     dlc: f64,
//     data: Vec<f64>,
//     flag: String,
// }

fn scale(x: f64, min: f64, max: f64) -> f64 {
    (x - min) / (max - min)
}

fn load_dataset(paths: Vec<&Path>) -> Result<TrainValSet, Error> {
    let (mut x_train, mut y_train) = (Vec::new(), Vec::new());
    let (mut x_test, mut y_test) = (Vec::new(), Vec::new());

    for path in paths {
        let (mut features, mut labels) = (Vec::new(), Vec::new());
        for dataset_type in ["train", "test"] {
            let cur_path = &format!("{}/{}.csv", path.display(), dataset_type);
            println!("Loading {}", cur_path);
            for record in csv::ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .from_path(cur_path)?
                .records()
            {
                let fields: StringRecord = record?;

                let _timestamp: f64 = match fields.get(0) {
                    Some(ts) => ts.parse().unwrap(),
                    None => 0.0,
                };

                let id: i32 = match fields.get(1) {
                    Some(id) => i32::from_str_radix(id, 16).unwrap_or(0),
                    None => 0,
                };

                let dlc: u8 = match fields.get(2) {
                    Some(ts) => ts.parse().unwrap(),
                    None => 0,
                };

                let mut data = Vec::new();
                for i in 0..dlc {
                    data.push(match fields.get(usize::from(i + 3)) {
                        Some(b) => {
                            scale(i32::from_str_radix(b, 16).unwrap_or(0) as f64, 0.0, 255.0)
                        }
                        None => 0.0,
                    });
                }

                let flag: bool = match fields.get(fields.len() - 1) {
                    Some(f) => f.chars().collect::<Vec<char>>()[0] == 'T',
                    None => false,
                };

                features.push([
                    scale(id as f64, 0.0, 65535.0),
                    scale(dlc as f64, 0.0, 8.0),
                    if dlc > 0 { data[0] } else { 0.0 },
                    if dlc > 1 { data[1] } else { 0.0 },
                    if dlc > 2 { data[2] } else { 0.0 },
                    if dlc > 3 { data[3] } else { 0.0 },
                    if dlc > 4 { data[4] } else { 0.0 },
                    if dlc > 5 { data[5] } else { 0.0 },
                    if dlc > 6 { data[6] } else { 0.0 },
                    if dlc > 7 { data[7] } else { 0.0 },
                ]);
                labels.push(flag);

                if dataset_type.contains("train") {
                    x_train.append(&mut features);
                    y_train.append(&mut labels);
                } else if dataset_type.contains("test") {
                    x_test.append(&mut features);
                    y_test.append(&mut labels);
                }
            }
        }
    }
    Ok((
        Dataset::new(Array2::from(x_train), Array1::from(y_train)),
        Dataset::new(Array2::from(x_test), Array1::from(y_test)),
    ))
}

fn main() -> linfa_svm::error::Result<()> {
    let args = Args::parse();
    let paths : Vec<&str> = args.paths.split(',').collect();
    
    let (train, valid) = match load_dataset(paths.iter().map(|p| Path::new(p)).collect()) {
        Ok(dataset) => dataset,
        Err(why) => panic!("Could not read file: {}", why),
    };

    println!("Loaded train dataset with shape: {:?}", train.records.shape());
    println!("Loaded validation dataset with shape: {:?}", valid.records.shape());

    println!("Training model...");
    let model = Svm::<f64, bool>::params().fit(&train)?;

    println!("Testing model...");
    let pred = model.predict(&valid);
    
    println!("Accuracy: {:?}", pred.confusion_matrix(&valid)?.accuracy());

    Ok(())
}
