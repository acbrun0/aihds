mod dataset;
mod model;
mod server;

use chrono::Utc;
use clap::Parser;
use linfa::prelude::*;
use serde::Deserialize;
use model::{svm, Ids, Packet};
use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
    thread, time,
};

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

#[derive(Deserialize)]
struct Config {
    window_size: usize,
    window_slide: u16
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let monitor: Option<Vec<u32>> = args.monitor.map(|monitor| {
        monitor
            .split(',')
            .collect::<Vec<&str>>()
            .iter()
            .map(|s| u32::from_str_radix(*s, 16).unwrap())
            .collect()
    });

    let baseline_size: usize = if monitor.is_some() {
        monitor.clone().unwrap().len() * 100
    } else {
        100000
    };

    let config: Config = match fs::read_to_string("config.toml") {
        Ok(file) => match toml::from_str(&file) {
            Ok(config) => config,
            Err(why) => panic!("Could not deserialize config file: {}", why)
        }
        Err(why) => panic!("Could not read config file: {}", why)
    };

    if let Some(paths) = args.extract_features {
        if let Some(modelpath) = args.model {
            let paths = paths.split(',').collect::<Vec<&str>>();
            let paths = paths.iter().map(Path::new).collect();
            let model = svm::load(Path::new(&modelpath)).expect("Could not load model");
            let scaler = bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
            let mut ids = Ids::new(
                Some(model),
                Some(scaler),
                config.window_size,
                config.window_slide,
                monitor,
            );
            println!("Loaded model from {}", modelpath);
            match dataset::packets_from_csv(paths) {
                Ok(packets) => {
                    ids.test(packets);
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
            let model = svm::load(Path::new(&modelpath)).expect("Could not load model");
            let scaler = bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
            ids = Ids::new(
                Some(model),
                Some(scaler),
                config.window_size,
                config.window_slide,
                monitor,
            );
            println!("Loaded model from {}", modelpath);
        } else {
            ids = Ids::new(None, None, config.window_size, config.window_slide, monitor);
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
        loop {
            match socket.read_frame() {
                Ok(frame) => {
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
                let mut ids = Ids::new(None, None, config.window_size, config.window_slide, monitor);
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
                                    thread::sleep(time::Duration::from_millis(50));
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
            let mut ids = Ids::new(None, None, config.window_size, config.window_slide, monitor);
            ids.train(None, Some(train_paths), baseline_size);
            if let Some(paths) = args.test {
                let test_paths: Vec<&str> = paths.split(',').collect();
                let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                match dataset::packets_from_csv(test_paths) {
                    Ok(packets) => {
                        let (real, pred, speed) = ids.test(packets);
                        let pred: Vec<bool> = pred.into_iter().map(|p| p.1).collect();
                        let mut tp = 0;
                        let mut fp = 0;
                        let mut tn = 0;
                        let mut fal_n = 0;
                        for (r, p) in real.iter().zip(pred.iter()) {
                            if *r {
                                if *p {
                                    tp += 1;
                                } else {
                                    fal_n += 1;
                                }
                            } else if *p {
                                fp += 1;
                            } else {
                                tn += 1;
                            }
                        }
                        println!("True positives: {}\nTrue negatives: {}\nFalse positives: {}\nFalse negatives: {}", tp, tn, fp, fal_n);
                        println!("Average packets/s: {}", speed);
                    }
                    Err(why) => panic!("Could not load test datasets: {}", why),
                }
            }
        }
    } else {
        // Load model
        println!("Loading model...");
        let modelpath = args.model.unwrap();
        match svm::load(Path::new(&modelpath)) {
            Ok(model) => {
                if let Some(url) = args.streaming {
                    if let Some(paths) = args.test {
                        let test_paths: Vec<&str> = paths.split(',').collect();
                        let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                        let scaler =
                            bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
                        match dataset::packets_from_csv(test_paths) {
                            Ok(packets) => {
                                let client = reqwest::Client::new();
                                let mut ids = Ids::new(
                                    Some(model),
                                    Some(scaler),
                                    config.window_size,
                                    config.window_slide,
                                    monitor,
                                );
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
                                            Ok(_) => (),
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
                        let scaler =
                            bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
                        let mut ids = Ids::new(
                            Some(model),
                            Some(scaler),
                            config.window_size,
                            config.window_slide,
                            monitor,
                        );
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
                                                println!(
                                                    "Could not communicate with server: {}",
                                                    why
                                                )
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
                    let scaler = bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
                    match dataset::packets_from_csv(test_paths) {
                        Ok(packets) => {
                            let (real, pred, speed) = Ids::new(
                                Some(model),
                                Some(scaler),
                                config.window_size,
                                config.window_slide,
                                monitor,
                            )
                            .test(packets);
                            let pred: Vec<bool> = pred.into_iter().map(|p| p.1).collect();
                            let mut tp = 0;
                            let mut fp = 0;
                            let mut tn = 0;
                            let mut fal_n = 0;
                            for (r, p) in real.iter().zip(pred.iter()) {
                                if *r {
                                    if *p {
                                        tp += 1;
                                    } else {
                                        fal_n += 1;
                                    }
                                } else if *p {
                                    fp += 1;
                                } else {
                                    tn += 1;
                                }
                            }
                            println!("True positives: {}\nTrue negatives: {}\nFalse positives: {}\nFalse negatives: {}", tp, tn, fp, fal_n);
                            println!("Average packets/s: {}", speed);
                        }
                        Err(why) => panic!("Could not load test datasets: {}", why),
                    }
                }
            }
            Err(why) => panic!("Could not load model: {}", why),
        }
    }
    Ok(())
}
