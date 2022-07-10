use crate::dataset;
use chrono::Utc;
use linfa::{prelude::*, DatasetBase};
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};
use socketcan::CANSocket;
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::Path,
    time::Instant,
};

pub type Features = [f64; 3];

#[derive(Serialize, Deserialize, Debug)]
pub struct Packet {
    timestamp: i64,
    id: u32,
    data: Vec<u8>,
    flag: bool,
}

impl Packet {
    pub fn new(timestamp: i64, id: u32, data: Vec<u8>, flag: bool) -> Packet {
        Packet {
            timestamp,
            id,
            data,
            flag,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Ids {
    model: Option<Svm<f64, bool>>,
    scaler: Option<Vec<(f64, f64)>>,
    window: Vec<Packet>,
    size: usize,
    slide: u16,
    counter: u16,
    monitor: Option<Vec<u32>>,
}

impl Ids {
    pub fn new(
        model: Option<Svm<f64, bool>>,
        scaler: Option<Vec<(f64, f64)>>,
        window_size: usize,
        window_slide: u16,
        monitor: Option<Vec<u32>>,
    ) -> Ids {
        Ids {
            model,
            scaler,
            window: Vec::with_capacity(window_size),
            size: window_size,
            slide: window_slide,
            counter: 0,
            monitor,
        }
    }

    pub fn load(path: &Path) -> Ids {
        let mut ids: Ids = bincode::deserialize(&fs::read(&path).unwrap()).unwrap();
        // Capacity is lost upon serialization
        ids.window = Vec::with_capacity(ids.size);
        ids
    }

    pub fn get_monitor(&self) -> &Option<Vec<u32>> {
        &self.monitor
    }

    pub fn train(
        &mut self,
        socket: Option<&CANSocket>,
        files: Option<Vec<&Path>>,
        baseline_len: usize,
    ) {
        let mut features: Vec<Features> = Vec::new();
        let mut labels = Vec::new();

        if let Some(socket) = socket {
            println!("Gathering baseline...");
            while features.len() < baseline_len {
                match socket.read_frame() {
                    Ok(frame) => {
                        let mut data = frame.data().to_vec();
                        while data.len() < 8 {
                            data.push(0);
                        }
                        let packet = Packet::new(
                            Utc::now().naive_local().timestamp_millis(),
                            frame.id(),
                            data,
                            false,
                        );
                        if self.window.len() < self.window.capacity() {
                            self.window.push(packet);
                        } else {
                            self.window.remove(0);
                            self.window.push(packet);
                            self.counter += 1;
                            if self.counter == self.slide {
                                if let Some(extracted) = self.extract_features() {
                                    features.push(extracted);
                                    labels.push(());
                                    self.counter = 0;
                                }
                            }
                        }
                        if features.len() as f32 % (baseline_len as f32 * 0.01) == 0.0 {
                            print!(
                                "{:.0}%\r",
                                features.len() as f32 / baseline_len as f32 * 100.0
                            );
                            io::stdout().flush().unwrap();
                        }
                    }
                    Err(why) => panic!("Could not read from socket: {}", why),
                }
            }
        } else if let Some(paths) = files {
            match dataset::packets_from_csv(paths) {
                Ok(packets) => {
                    for packet in packets {
                        if let Some(filter) = &self.monitor {
                            if !filter.contains(&packet.id) {
                                continue;
                            }
                        }
                        if self.window.len() < self.window.capacity() {
                            self.window.push(packet);
                        } else {
                            self.window.remove(0);
                            self.window.push(packet);
                            self.counter += 1;
                            if self.counter == self.slide {
                                if let Some(extracted) = self.extract_features() {
                                    features.push(extracted);
                                    labels.push(());
                                    self.counter = 0;
                                }
                            }
                        }
                    }
                    println!("Training with {} features", features.len());
                }
                Err(why) => panic!("Could not load datasets: {}", why),
            }
        }

        let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
            // .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist", "HammingDistBytes", "GapBytes"]);
            .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist"]);
            // .with_feature_names(vec!["AvgTime", "HammingDist"]);
        let scaler = dataset::normalize(&mut dataset, &None);

        match Svm::<f64, _>::params()
            .gaussian_kernel(1.0)
            .nu_weight(0.001)
            .fit(&dataset)
        {
            Ok(model) => {
                self.window.clear();
                self.counter = 0;
                self.model = Some(model);
                self.scaler = scaler;
                match std::fs::create_dir_all(Path::new("models")) {
                    Ok(()) => {
                        fs::write("models/ids", bincode::serialize(self).unwrap())
                            .expect("Could not save model");
                    }
                    Err(why) => panic!("Could not create models directory: {}", why),
                }
                match std::fs::create_dir_all(Path::new("features")) {
                    Ok(()) => {
                        dataset::write_features_unsupervised(
                            Path::new("features/train.csv"),
                            &dataset,
                        )
                        .expect("Could not save train features");
                    }
                    Err(why) => panic!("Could not create features directory: {}", why),
                }
            }
            Err(why) => panic!("Could not train model: {}", why),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn feature_file(
        &mut self,
        packets: Vec<Packet>,
    ) -> DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<bool>, Dim<[usize; 1]>>,
    > {
        let mut features = Vec::new();
        let mut labels = Vec::new();

        for packet in packets {
            if let Some(filter) = &self.monitor {
                if !filter.contains(&packet.id) {
                    continue;
                }
            }
            if let Some(result) = self.push(packet) {
                features.push(result.0);
                labels.push(self.window.iter().any(|p| p.flag));
            }
        }

        Dataset::new(Array2::from(features), Array1::from(labels)).with_feature_names(vec![
            "AvgTime",
            "Entropy",
            "HammingDist",
            // "HammingDistBytes",
            // "GapBytes",
            "Label",
        ])
    }

    #[allow(clippy::type_complexity)]
    pub fn test(
        &mut self,
        packets: Vec<Packet>,
    ) -> (Vec<bool>, Vec<(Features, bool, (i64, i64))>, f32) {
        let mut predictions = Vec::new();
        let mut real = Vec::new();
        let mut features = Vec::new();
        let n_packets = packets.len();

        let start = Instant::now();
        for packet in packets {
            if let Some(result) = self.push(packet) {
                features.push(result.0);
                predictions.push(result);
                real.push(self.window.iter().any(|p| p.flag));
            }
        }
        let duration = start.elapsed().as_secs_f32();
        match std::fs::create_dir_all(Path::new("features")) {
            Ok(()) => {
                match dataset::write_features(
                    Path::new("features/test.csv"),
                    &Dataset::new(
                        Array2::from(features),
                        Array1::from(
                            predictions
                                .iter()
                                .zip(real.iter())
                                .map(|(p, r)| {
                                    if *r {
                                        if p.1 {
                                            0
                                        } else {
                                            3
                                        }
                                    } else if p.1 {
                                        2
                                    } else {
                                        1
                                    }
                                })
                                .collect::<Vec<u8>>(),
                        ),
                    )
                    .with_feature_names(vec![
                        "AvgTime",
                        "Entropy",
                        "HammingDist",
                        // "HammingDistBytes",
                        // "GapBytes",
                        "Label",
                    ]),
                ) {
                    Ok(_) => (),
                    Err(why) => println!("Could not save test features: {}", why),
                }
            }
            Err(why) => panic!("Could not create features directory: {}", why),
        }

        (real, predictions, n_packets as f32 / duration)
    }

    pub fn push(&mut self, packet: Packet) -> Option<(Features, bool, (i64, i64))> {
        if let Some(filter) = &self.monitor {
            if !filter.contains(&packet.id) {
                return None;
            }
        }
        let mut prediction: Option<(Features, bool, (i64, i64))> = None;
        if self.window.len() < self.window.capacity() {
            self.window.push(packet);
        } else {
            self.window.remove(0);
            self.window.push(packet);
            self.counter += 1;
            if self.counter == self.slide {
                if let Some(pred) = self.predict() {
                    prediction = Some((
                        pred.0,
                        pred.1,
                        (
                            self.window[0].timestamp,
                            self.window[self.window.capacity() - 1].timestamp,
                        ),
                    ));
                    self.counter = 0;
                }
            }
        };
        prediction
    }

    fn predict(&self) -> Option<(Features, bool)> {
        if let Some(scaler) = &self.scaler {
            if let Some(model) = &self.model {
                let mut si = 0;
                if let Some(features) = self.extract_features() {
                    let features: Features = features
                        .iter()
                        .map(|f| {
                            si += 1;
                            (f - scaler[si - 1].0) / (scaler[si - 1].1 - scaler[si - 1].0)
                        })
                        .collect::<Vec<f64>>()
                        .as_slice()
                        .try_into()
                        .unwrap();
                    // Positive class is normal
                    // Negative class is anomaly
                    Some((features, !model.predict(Array1::from(features.to_vec()))))
                } else {
                    None
                }
            } else {
                panic!("IDS does not have a model.");
            }
        } else {
            panic!("IDS does not have a scaler.");
        }
    }

    fn extract_features(&self) -> Option<Features> {
        let mut feat = HashMap::new();
        let mut avg_time = Vec::new();
        let mut entropy = Vec::new();
        // let mut hamming: Vec<f64> = Vec::new();
        let mut hamming_bytes = Vec::new();
        // let mut diff_bytes = Vec::new();

        for p in &self.window {
            let val = feat.entry(&p.id).or_insert((Vec::new(), Vec::new()));
            val.0.push(p.timestamp);
            val.1.push(&p.data);
        }

        if !feat.is_empty() {
            for val in feat.values() {
                // Calculate average arriving interval
                let mut interval = Vec::new();
                if val.0.len() > 1 {
                    interval = val.0.windows(2).map(|w| w[1] - w[0]).collect::<Vec<i64>>();
                } else {
                    interval.push(0);
                }
                avg_time.push(if !val.0.is_empty() {
                    interval.iter().sum::<i64>() as f64 / val.0.len() as f64
                } else {
                    0.0
                });

                if val.1.len() > 1 {
                    // // Calculate average Hamming distance
                    // hamming.push(
                    //     val.1
                    //         .windows(2)
                    //         .map(|b| {
                    //             let mut count: i16 = 0;
                    //             let bitstring0 =
                    //                 b[0].iter().fold(String::new(), |mut string, b| {
                    //                     string.push_str(format!("{:08b}", b).as_str());
                    //                     string
                    //                 });
                    //             let bitstring1 =
                    //                 b[1].iter().fold(String::new(), |mut string, b| {
                    //                     string.push_str(format!("{:08b}", b).as_str());
                    //                     string
                    //                 });
                    //             for (b0, b1) in bitstring0.chars().zip(bitstring1.chars()) {
                    //                 if b0 != b1 {
                    //                     count += 1;
                    //                 }
                    //             }
                    //             count
                    //         })
                    //         .collect::<Vec<i16>>()
                    //         .iter()
                    //         .sum::<i16>() as f64
                    //         / (val.1.len()) as f64,
                    // );
                    // Calculate entropy
                    for i in 0..8 {
                        // Get ith byte from each ocurrence
                        let byteset: Vec<u8> = val.1.iter().map(|bytes| bytes[i]).collect();
                        // Count number of times each byte ocurrs
                        let mut bytemap = HashMap::new();
                        for byte in byteset.iter() {
                            let entry = bytemap.entry(byte).or_insert(0);
                            *entry += 1;
                        }
                        // Get each byte's probability of ocurring
                        let mut byteprobs = Vec::new();
                        for count in bytemap.values() {
                            byteprobs.push(*count as f64 / byteset.len() as f64)
                        }
                        // Calculate ith position's entropy
                        entropy.push(0.0 - byteprobs.iter().map(|p| p * p.log2()).sum::<f64>());
                    }
                    // Calculate hamming distance byte-wise
                    hamming_bytes.push(
                        val.1
                            .windows(2)
                            .map(|b| {
                                let mut count: i16 = 0;
                                for (b1, b2) in b[0].iter().zip(b[1]) {
                                    if b1 != b2 {
                                        count += 1;
                                    }
                                }
                                count
                            })
                            .collect::<Vec<i16>>()
                            .iter()
                            .sum::<i16>() as f64
                            / (val.1.len()) as f64,
                    );
                    // // Calculate gap between bytes
                    // diff_bytes.push(
                    //     val.1
                    //         .windows(2)
                    //         .map(|b| {
                    //             let mut gaps = Vec::new();
                    //             for (b1, b2) in b[0].iter().zip(b[1]) {
                    //                 gaps.push(*b2 as i16 - *b1 as i16);
                    //             }
                    //             gaps.iter().sum::<i16>() as f64 / gaps.len() as f64
                    //         })
                    //         .collect::<Vec<f64>>()
                    //         .iter()
                    //         .sum::<f64>()
                    //         / (val.1.len()) as f64,
                    // );
                }
            }

            Some([
                avg_time.iter().sum::<f64>() / avg_time.len() as f64,
                entropy.iter().sum::<f64>() / entropy.len() as f64,
                // hamming.iter().sum::<f64>() / hamming.len() as f64,
                hamming_bytes.iter().sum::<f64>() / hamming_bytes.len() as f64,
                // diff_bytes.iter().sum::<f64>() / diff_bytes.len() as f64,
            ])
        } else {
            None
        }
    }
}
