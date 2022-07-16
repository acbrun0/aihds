#![warn(missing_docs)]

//! **Automated and Intelligent Hacking Detection System**  
//! A light-weight One Class SVM-based Intrusion Detection System for the Controller Area Network (CAN)
//!
//! This crate provides a machine-learning based intrusion detection system capable of detecting known and unkown CAN attacks.
//! Features extracted from an attack-free basline are used to train the model, which then is able of analysing real-time traffic. The extrated features are:
//! - Average packet frequency
//! - Average network entropy
//! - Average Hamming distance between consecutive packets

mod dataset;
mod ids;
mod server;

use chrono::Utc;
use clap::Parser;
use csv::Writer;
use ids::{Ids, Packet};
use linfa::prelude::*;
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
    thread, time,
};

/// Defines the set of CLI options.
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
    model: Option<String>,
    /// Extracts features to CSV files
    #[clap(long)]
    extract_features: Option<String>,
    /// Perform grid search optimization on SVM
    #[clap(long)]
    grid_search: bool,
    /// Join features into a single file
    #[clap(long)]
    join: bool,
    /// Run model in streaming mode
    #[clap(long)]
    streaming: Option<String>,
    /// IDs to monitor
    #[clap(long)]
    monitor: Option<String>,
    /// Run IDS in live mode
    #[clap(long)]
    live: bool,
}

/// Describes the configuration file.
#[derive(Deserialize)]
struct Config {
    window: Window,
}

/// Represents a IDS window.
#[derive(Deserialize)]
struct Window {
    /// The size of the window.
    size: usize,
    /// The number of packets that enter and leave the window before feature extraction is performed.
    slide: u16,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let mut monitor: Option<Vec<u32>> = args.monitor.map(|monitor| {
        monitor
            .split(',')
            .collect::<Vec<&str>>()
            .iter()
            .map(|s| u32::from_str_radix(*s, 16).unwrap())
            .collect::<Vec<u32>>()
    });
    let baseline_size = if let Some(ref mut monitor) = monitor {
        monitor.sort_unstable();
        monitor.dedup();
        println!("Monitoring {} IDs", monitor.len());
        monitor.len() * 1000
    } else {
        1000000
    };

    let config: Config = match fs::read_to_string("config.toml") {
        Ok(file) => match toml::from_str(&file) {
            Ok(config) => config,
            Err(why) => panic!("Could not deserialize config file: {}", why),
        },
        Err(why) => panic!("Could not read config file: {}", why),
    };

    if let Some(paths) = args.extract_features {
        if let Some(modelpath) = args.model {
            let paths = paths.split(',').collect::<Vec<&str>>();
            let paths = paths.iter().map(Path::new).collect();
            let mut ids = Ids::load(Path::new(&modelpath));
            println!("Loaded model from {}", modelpath);
            match dataset::packets_from_csv(paths) {
                Ok(packets) => {
                    let dataset = ids.feature_set(packets);
                    let mut wtr = Writer::from_path(Path::new("features/extracted.csv")).unwrap();
                    match wtr.write_record(dataset.feature_names()) {
                        Ok(()) => {
                            for (record, target) in
                                dataset.records.outer_iter().zip(dataset.targets.clone())
                            {
                                wtr.write_record(&[
                                    record[0].to_string(),
                                    record[1].to_string(),
                                    record[2].to_string(),
                                    record[3].to_string(),
                                    record[4].to_string(),
                                    target.to_string(),
                                ])
                                .unwrap();
                            }
                            wtr.flush().unwrap();
                        }
                        Err(why) => panic!(
                            "Could not write to {}: {}",
                            Path::new("features/extracted.csv").display(),
                            why
                        ),
                    }
                    return Ok(());
                }
                Err(why) => panic!("Could not load dataset: {}", why),
            }
        } else {
            panic!("Please specify a path to the model using --model")
        }
    }

    if args.live {
        let mut ids;
        let socket = server::open_socket("can0", &monitor);
        let client = reqwest::Client::new();
        let mut last_attack = time::Instant::now();

        if let Some(modelpath) = args.model {
            ids = Ids::load(Path::new(&modelpath));
            println!("Loaded model from {}", modelpath);
            if let Some(monitor) = ids.get_monitor() {
                println!("Monitoring: {:X?}", monitor);
            } else {
                println!("Monitoring all IDs.");
            }
        } else {
            ids = Ids::new(None, None, config.window.size, config.window.slide, monitor);
            ids.train(Some(&socket), None, baseline_size);
            println!("Training complete");
        }

        println!("Analysing network...");
        let mut log = match File::create("models/log.csv") {
            Ok(file) => file,
            Err(why) => panic!("Could not create log file: {}", why),
        };
        log.write_all(b"AvgTime,Entropy,HammingDist,Label\n")
            .expect("Unable to write to log");

        let mut speed_file = match File::create("models/speed.log") {
            Ok(file) => file,
            Err(why) => panic!("Could not create log file: {}", why),
        };
        let mut n_packets = 0;
        let mut start_time = time::Instant::now();
        loop {
            match socket.read_frame() {
                Ok(frame) => {
                    n_packets += 1;

                    let mut data = frame.data().to_vec();
                    while data.len() < 8 {
                        data.push(0);
                    }

                    if let Some(result) = ids.push(Packet::new(
                        Utc::now().naive_local().timestamp_millis(),
                        frame.id(),
                        data,
                        false,
                    )) {
                        // Measure system speed
                        if n_packets == 1000 {
                            let duration = start_time.elapsed().as_secs_f64();
                            speed_file
                                .write_all(format!("{}\n", 1000_f64 / duration).as_bytes())
                                .expect("Unable to write to log");
                            n_packets = 0;
                            start_time = time::Instant::now();
                        }

                        log.write_all(
                            format!(
                                "{},{},{},{}\n",
                                result.0[0], result.0[1], result.0[2], result.1
                            )
                            .as_bytes(),
                        )
                        .expect("Unable to write to log");
                        if let Some(url) = &args.streaming {
                            match server::post(
                                &client,
                                url,
                                result.0.iter().map(|f| *f as f32).collect(),
                                &result.1,
                                if result.1 {
                                    Some(format!(
                                        "Found attack inside window [{}, {}]",
                                        result.2 .0, result.2 .1
                                    ))
                                } else {
                                    None
                                },
                            )
                            .await
                            {
                                Ok(_) => (),
                                Err(why) => {
                                    println!("Could not communicate with server: {}", why)
                                }
                            }
                        } else if result.1 {
                            if time::Instant::now().duration_since(last_attack).as_millis() < 1000 {
                                print!("Attack detected\r");
                                io::stdout().flush().unwrap();
                            }
                            last_attack = time::Instant::now();
                        } else if time::Instant::now().duration_since(last_attack).as_millis()
                            > 1000
                        {
                            // Clear line
                            print!("               \r");
                            io::stdout().flush().unwrap();
                        }
                    // Measure system speed
                    } else if n_packets == 1000 {
                        let duration = start_time.elapsed().as_secs_f64();
                        speed_file
                            .write_all(format!("{}\n", 1000_f64 / duration).as_bytes())
                            .expect("Unable to write to log");
                        n_packets = 0;
                        start_time = time::Instant::now();
                    }
                }
                Err(why) => panic!("Could not read frame: {}", why),
            }
        }
    } else if args.model.is_none() {
        if let Some(url) = args.streaming {
            if let Some(paths) = args.train {
                let train_paths: Vec<&str> = paths.split(',').collect();
                let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
                let mut ids =
                    Ids::new(None, None, config.window.size, config.window.slide, monitor);
                ids.train(None, Some(train_paths), baseline_size);
                if let Some(paths) = args.test {
                    let test_paths: Vec<&str> = paths.split(',').collect();
                    let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                    match dataset::packets_from_csv(test_paths) {
                        Ok(packets) => {
                            let client = reqwest::Client::new();
                            for packet in packets {
                                if let Some(result) = ids.push(packet) {
                                    // Sleep to allow chart animation to complete
                                    thread::sleep(time::Duration::from_millis(100));
                                    match server::post(
                                        &client,
                                        &url,
                                        result.0.to_vec().iter().map(|v| *v as f32).collect(),
                                        &result.1,
                                        if result.1 {
                                            Some(format!(
                                                "Found attack inside window [{}, {}]",
                                                result.2 .0, result.2 .1
                                            ))
                                        } else {
                                            None
                                        },
                                    )
                                    .await
                                    {
                                        Ok(_) => (),
                                        Err(why) => {
                                            panic!("Could not communicate with server: {}", why)
                                        }
                                    }
                                }
                            }
                        }
                        Err(why) => panic!("Could not load test datasets: {}", why),
                    }
                }
            } else {
                panic!("Did not provide model or datasets to train one.")
            }
        } else if let Some(paths) = args.train {
            let train_paths: Vec<&str> = paths.split(',').collect();
            let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
            let mut ids = Ids::new(None, None, config.window.size, config.window.slide, monitor);
            ids.train(None, Some(train_paths), baseline_size);
            if let Some(paths) = args.test {
                let test_paths: Vec<&str> = paths.split(',').collect();
                let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                match dataset::packets_from_csv(test_paths) {
                    Ok(packets) => {
                        let (real, pred, speed) = ids.test(packets);
                        let pred: Vec<bool> = pred.into_iter().map(|p| p.1).collect();
                        let mut tp: f64 = 0.0;
                        let mut fp: f64 = 0.0;
                        let mut tn: f64 = 0.0;
                        let mut fal_n: f64 = 0.0;
                        for (r, p) in real.iter().zip(pred.iter()) {
                            if *r {
                                if *p {
                                    tp += 1.0;
                                } else {
                                    fal_n += 1.0;
                                }
                            } else if *p {
                                fp += 1.0;
                            } else {
                                tn += 1.0;
                            }
                        }
                        let precision = tp / (tp + fp);
                        let recall = tp / (tp + fal_n);
                        println!("True positives: {}\nTrue negatives: {}\nFalse positives: {}\nFalse negatives: {}\n", tp, tn, fp, fal_n);
                        println!("False negative rate: {:.3}%", fal_n / (tp + fal_n) * 100.0);
                        println!(
                            "Error rate: {:.3}%",
                            (fp + fal_n) / (tp + tn + fp + fal_n) * 100.0
                        );
                        println!("Precision: {:.3}%", precision * 100.0);
                        println!("Recall: {:.3}%", recall * 100.0);
                        println!(
                            "F1-score: {:.3}%\n",
                            2.0 * (precision * recall) / (precision + recall) * 100.0
                        );
                        println!("Average packets/s: {}", speed);
                    }
                    Err(why) => panic!("Could not load test datasets: {}", why),
                }
            }
        }
    } else {
        // Load model
        let modelpath = args.model.unwrap();
        if let Some(url) = args.streaming {
            if let Some(paths) = args.test {
                let test_paths: Vec<&str> = paths.split(',').collect();
                let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                match dataset::packets_from_csv(test_paths) {
                    Ok(packets) => {
                        let client = reqwest::Client::new();
                        println!("Loading model...");
                        let mut ids = Ids::load(Path::new(&modelpath));
                        if let Some(monitor) = ids.get_monitor() {
                            println!("Monitoring: {:X?}", monitor);
                        } else {
                            println!("Monitoring all IDs.");
                        }
                        for packet in packets {
                            if let Some(result) = ids.push(packet) {
                                match server::post(
                                    &client,
                                    &url,
                                    result.0.to_vec().iter().map(|v| *v as f32).collect(),
                                    &result.1,
                                    if result.1 {
                                        Some(format!(
                                            "Found attack inside window [{}, {}]",
                                            result.2 .0, result.2 .1
                                        ))
                                    } else {
                                        None
                                    },
                                )
                                .await
                                {
                                    Ok(_) => {
                                        // Sleep to allow chart animation to complete
                                        thread::sleep(time::Duration::from_millis(100))
                                    }
                                    Err(why) => {
                                        panic!("Could not communicate with server: {}", why)
                                    }
                                }
                            }
                        }
                    }
                    Err(why) => panic!("Could not load test dataset: {}", why),
                }
            } else {
                let socket = server::open_socket("can0", &monitor);
                let client = reqwest::Client::new();
                let mut ids = Ids::load(Path::new(&modelpath));
                if let Some(monitor) = ids.get_monitor() {
                    println!("Monitoring: {:X?}", monitor);
                } else {
                    println!("Monitoring all IDs.");
                }
                loop {
                    match socket.read_frame() {
                        Ok(frame) => {
                            if let Some(result) = ids.push(Packet::new(
                                Utc::now().naive_local().timestamp_millis(),
                                frame.id(),
                                frame.data().to_vec(),
                                false,
                            )) {
                                match server::post(
                                    &client,
                                    &url,
                                    result.0.iter().map(|f| *f as f32).collect(),
                                    &result.1,
                                    if result.1 {
                                        Some(format!(
                                            "Found attack inside window [{}, {}]",
                                            result.2 .0, result.2 .1
                                        ))
                                    } else {
                                        None
                                    },
                                )
                                .await
                                {
                                    Ok(_) => (),
                                    Err(why) => {
                                        println!("Could not communicate with server: {}", why)
                                    }
                                }
                            }
                        }
                        Err(why) => panic!("Could not read frame: {}", why),
                    }
                }
            }
        } else if let Some(paths) = args.test {
            let test_paths: Vec<&str> = paths.split(',').collect();
            let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
            match dataset::packets_from_csv(test_paths) {
                Ok(packets) => {
                    let mut ids = Ids::load(Path::new(&modelpath));
                    if let Some(monitor) = ids.get_monitor() {
                        println!("Monitoring: {:X?}", monitor);
                    } else {
                        println!("Monitoring all IDs.");
                    }
                    let (real, pred, speed) = ids.test(packets);
                    let pred: Vec<bool> = pred.into_iter().map(|p| p.1).collect();
                    let mut tp: f64 = 0.0;
                    let mut fp: f64 = 0.0;
                    let mut tn: f64 = 0.0;
                    let mut fal_n: f64 = 0.0;
                    for (r, p) in real.iter().zip(pred.iter()) {
                        if *r {
                            if *p {
                                tp += 1.0;
                            } else {
                                fal_n += 1.0;
                            }
                        } else if *p {
                            fp += 1.0;
                        } else {
                            tn += 1.0;
                        }
                    }
                    let precision = tp / (tp + fp);
                    let recall = tp / (tp + fal_n);
                    println!("True positives: {}\nTrue negatives: {}\nFalse positives: {}\nFalse negatives: {}\n", tp, tn, fp, fal_n);
                    println!("False negative rate: {:.3}%", fal_n / (tp + fal_n) * 100.0);
                    println!(
                        "Error rate: {:.3}%",
                        (fp + fal_n) / (tp + tn + fp + fal_n) * 100.0
                    );
                    println!("Precision: {:.3}%", precision * 100.0);
                    println!("Recall: {:.3}%", recall * 100.0);
                    println!(
                        "F1-score: {:.3}%\n",
                        2.0 * (precision * recall) / (precision + recall) * 100.0
                    );
                    println!("Average packets/s: {}", speed);
                }
                Err(why) => panic!("Could not load test datasets: {}", why),
            }
        }
    }
    Ok(())
}
