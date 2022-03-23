mod dataset;
mod model;
mod server;

use chrono::Utc;
use clap::Parser;
use linfa::prelude::*;
use model::{svm, Ids, Packet};
use std::{fs, io, io::Write, path::Path};

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

const BASELINE_SIZE: usize = 100000;
const WINDOW_SIZE: usize = 500;
const WINDOW_SLIDE: u16 = 50;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let monitor = args.monitor.map(|monitor| {
        monitor
            .split(',')
            .collect::<Vec<&str>>()
            .iter()
            .map(|s| String::from(*s))
            .collect()
    });

    if args.extract_features {
        match std::fs::create_dir_all(Path::new("features")) {
            Ok(_) => (),
            Err(why) => panic!("Could not create features directory: {}", why),
        };
        if let Some(paths) = args.train {
            let paths = paths.split(',').collect::<Vec<&str>>();
            let paths = paths.iter().map(Path::new).collect();
            if args.join {
                match dataset::load(paths, None, &monitor) {
                    Ok((mut dataset, _)) => {
                        match dataset::write_features(
                            Path::new(&format!(
                                "{}/features.csv",
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
                    match dataset::load(vec![path], None, &monitor) {
                        Ok((mut dataset, _)) => {
                            match dataset::write_features(
                                Path::new(&format!(
                                    "features/{}.csv",
                                    path.file_stem().unwrap().to_str().unwrap()
                                )),
                                &mut dataset,
                                false,
                            ) {
                                Ok(_) => (),
                                Err(why) => {
                                    println!("Could not write features: {}", why)
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
                match dataset::load(paths, scaler, &monitor) {
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
                    match dataset::load(vec![path], scaler_copy, &monitor) {
                        Ok((mut dataset, scaler)) => {
                            scaler_copy = Some(scaler);
                            match dataset::write_features(
                                Path::new(&format!(
                                    "features/{}.csv",
                                    path.file_stem().unwrap().to_str().unwrap()
                                )),
                                &mut dataset,
                                args.libsvm,
                            ) {
                                Ok(_) => (),
                                Err(why) => {
                                    println!("Could not write features: {}", why)
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

    // if args.grid_search {
    //     match args.train {
    //         Some(paths) => {
    //             let train_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
    //             let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
    //             if let Some(paths) = args.test {
    //                 let test_paths: Vec<&str> = paths.split(',').collect::<Vec<&str>>();
    //                 let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
    //                 match dataset::load(train_paths, None, &monitor) {
    //                     Ok((train_dataset, scaler)) => {
    //                         match dataset::load(test_paths, Some(scaler), &monitor) {
    //                             Ok((test_dataset, _)) => {
    //                                 match svm::grid_search(&train_dataset, &test_dataset) {
    //                                     Ok(results) => println!("{:#?}", results),
    //                                     Err(why) => {
    //                                         panic!("Could not perform grid search: {}", why)
    //                                     }
    //                                 }
    //                             }
    //                             Err(why) => panic!("Could not load test datasets: {}", why),
    //                         }
    //                     }
    //                     Err(why) => panic!("Could not load train datasets: {}", why),
    //                 }
    //             }
    //         }
    //         None => panic!("Did not provide path to train datasets."),
    //     }
    // } else
    if args.live {
        let mut baseline: Vec<Packet> = Vec::with_capacity(BASELINE_SIZE);
        match server::open_socket("can0") {
            Ok(socket) => {
                println!("Gathering baseline...");
                while baseline.len() <= BASELINE_SIZE {
                    match socket.read_frame() {
                        Ok(frame) => {
                            baseline.push(Packet::new(
                                Utc::now().naive_local().timestamp_nanos(),
                                frame.id(),
                                frame.data().to_vec(),
                                false,
                            ));
                            if baseline.len() as f32 % (BASELINE_SIZE as f32 * 0.05) == 0.0 {
                                print!(
                                    "{:.0}%\r",
                                    baseline.len() as f32 / BASELINE_SIZE as f32 * 100.0
                                );
                                io::stdout().flush().unwrap();
                            }
                        }
                        Err(why) => panic!("Could not read frame: {}", why),
                    }
                }
                let mut ids = Ids::new(None, None, WINDOW_SIZE, WINDOW_SLIDE, None);
                println!("Training model...");
                ids.train(baseline);
                println!("Training complete");
                println!("Analysing network...");
                loop {
                    match socket.read_frame() {
                        Ok(frame) => {
                            if let Some(result) = ids.push(Packet::new(
                                Utc::now().naive_local().timestamp_nanos(),
                                frame.id(),
                                frame.data().to_vec(),
                                false,
                            )) {
                                if result.1 {
                                    println!("{:?}", result);
                                }
                            }
                        }
                        Err(why) => panic!("Could not read frame: {}", why),
                    }
                }
            }
            Err(why) => panic!("Could not open socket: {}", why),
        }
    } else if args.model.is_none() {
        if let Some(url) = args.streaming {
            if let Some(paths) = args.train {
                let train_paths: Vec<&str> = paths.split(',').collect();
                let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
                match dataset::packets_from_csv(train_paths) {
                    Ok(packets) => {
                        let mut ids = Ids::new(None, None, WINDOW_SIZE, WINDOW_SLIDE, None);
                        ids.train(packets);
                        if let Some(paths) = args.test {
                            let test_paths: Vec<&str> = paths.split(',').collect();
                            let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                            match dataset::packets_from_csv(test_paths) {
                                Ok(packets) => {
                                    let client = reqwest::Client::new();
                                    for packet in packets {
                                        if let Some(result) = ids.push(packet) {
                                            match server::post(
                                                &client,
                                                &url,
                                                result
                                                    .0
                                                    .to_vec()
                                                    .iter()
                                                    .map(|v| *v as f32)
                                                    .collect(),
                                                &result.1,
                                            )
                                            .await
                                            {
                                                Ok(_) => (),
                                                Err(why) => panic!(
                                                    "Could not communicate with server: {}",
                                                    why
                                                ),
                                            }
                                        }
                                    }
                                }
                                Err(why) => panic!("Could not load test datasets: {}", why),
                            }
                        }
                    }
                    Err(why) => panic!("Could not load train datasets: {}", why),
                }
            } else {
                panic!("Did not provide model or datasets to train one.")
            }
        } else if let Some(paths) = args.train {
            let train_paths: Vec<&str> = paths.split(',').collect();
            let train_paths: Vec<&Path> = train_paths.iter().map(Path::new).collect();
            match dataset::packets_from_csv(train_paths) {
                Ok(packets) => {
                    let mut ids = Ids::new(None, None, WINDOW_SIZE, WINDOW_SLIDE, None);
                    ids.train(packets);
                    if let Some(paths) = args.test {
                        let test_paths: Vec<&str> = paths.split(',').collect();
                        let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                        match dataset::packets_from_csv(test_paths) {
                            Ok(packets) => {
                                let (real, pred) = ids.test(packets);
                                let pred: Vec<bool> = pred
                                    .into_iter()
                                    .map(|p| p.1)
                                    .collect();
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
                            }
                            Err(why) => panic!("Could not load test datasets: {}", why),
                        }
                    }
                }
                Err(why) => panic!("Could not load train datasets: {}", why),
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
                                let mut ids =
                                    Ids::new(Some(model), scaler, WINDOW_SIZE, WINDOW_SLIDE, None);
                                for packet in packets {
                                    if let Some(result) = ids.push(packet) {
                                        match server::post(
                                            &client,
                                            &url,
                                            result.0.to_vec().iter().map(|v| *v as f32).collect(),
                                            &result.1,
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
                        match server::open_socket("can0") {
                            Ok(socket) => {
                                let client = reqwest::Client::new();
                                let scaler =
                                    bincode::deserialize(&fs::read("models/scaler").unwrap())
                                        .unwrap();
                                let mut ids = Ids::new(Some(model), Some(scaler), 200, 50, None);
                                loop {
                                    match socket.read_frame() {
                                        Ok(frame) => {
                                            if let Some(result) = ids.push(Packet::new(
                                                Utc::now().naive_local().timestamp_nanos(),
                                                frame.id(),
                                                frame.data().to_vec(),
                                                false,
                                            )) {
                                                // println!("{:?}", result);
                                                match server::post(
                                                    &client,
                                                    &url,
                                                    result.0.iter().map(|f| *f as f32).collect(),
                                                    &result.1,
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
                            Err(why) => panic!("Could not open socket: {}", why),
                        }
                    }
                } else if let Some(paths) = args.test {
                    let test_paths: Vec<&str> = paths.split(',').collect();
                    let test_paths: Vec<&Path> = test_paths.iter().map(Path::new).collect();
                    let scaler = bincode::deserialize(&fs::read("models/scaler").unwrap()).unwrap();
                    match dataset::packets_from_csv(test_paths) {
                        Ok(packets) => {
                            let (real, pred) = Ids::new(
                                Some(model),
                                Some(scaler),
                                WINDOW_SIZE,
                                WINDOW_SLIDE,
                                None,
                            )
                            .test(packets);
                            let pred: Vec<bool> = pred
                                .into_iter()
                                .map(|p| p.1)
                                .collect();
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
