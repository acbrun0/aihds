use crate::dataset;
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use std::{collections::HashMap, fs, path::Path, time::Instant};

pub type Features = [f64; 3];

#[derive(Debug)]
pub struct Packet {
    timestamp: i64,
    id: String,
    data: Vec<u8>,
    flag: bool,
}

impl Packet {
    pub fn new(timestamp: i64, id: String, data: Vec<u8>, flag: bool) -> Packet {
        Packet {
            timestamp,
            id,
            data,
            flag,
        }
    }
}

pub struct Ids {
    model: Option<Svm<f64, bool>>,
    scaler: Option<Vec<(f64, f64)>>,
    window: Vec<Packet>,
    slide: u16,
    counter: u16,
    monitor: Option<Vec<String>>,
}

impl Ids {
    pub fn new(
        model: Option<Svm<f64, bool>>,
        scaler: Option<Vec<(f64, f64)>>,
        window_size: usize,
        window_slide: u16,
        monitor: Option<Vec<String>>,
    ) -> Ids {
        Ids {
            model,
            scaler,
            window: Vec::with_capacity(window_size),
            slide: window_slide,
            counter: 0,
            monitor,
        }
    }

    pub fn train(&mut self, packets: Vec<Packet>) {
        let mut features: Vec<Features> = Vec::new();
        let mut labels = Vec::new();

        for packet in packets {
            if self.window.len() < self.window.capacity() {
                self.window.push(packet);
            } else {
                self.window.remove(0);
                self.window.push(packet);
                self.counter += 1;
                if self.counter == self.slide {
                    if let Some(extracted) = self.extract_features() {
                        let mut feat: Features = [0.0, 0.0, 0.0];
                        for (i, f) in extracted.iter().enumerate() {
                            feat[i] = *f;
                        }
                        features.push(feat);
                        labels.push(());
                        self.counter = 0;
                    }
                }
            }
        }

        let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
            .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist"]);
        self.scaler = dataset::normalize_unsupervised(&mut dataset, &None);

        match Svm::<f64, _>::params()
            .gaussian_kernel(1.0)
            // .polynomial_kernel(0.0, 3.0)
            .nu_weight(0.001)
            .fit(&dataset)
        {
            Ok(model) => {
                self.window.clear();
                self.counter = 0;
                match std::fs::create_dir_all(Path::new("models")) {
                    Ok(_) => {
                        dataset::write_features_unsupervised(
                            Path::new("models/train.csv"),
                            &dataset,
                            false,
                        )
                        .expect("Could not save train features");
                        fs::write("models/scaler", bincode::serialize(&self.scaler).unwrap())
                            .expect("Could not save scaler.");
                        save(&model, Path::new("models/svm")).expect("Could not save model");
                    }
                    Err(why) => panic!("Could not create models directory: {}", why),
                };
                self.model = Some(model);
            }
            Err(why) => panic!("Could not train model: {}", why),
        }
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

        match dataset::write_features(
            Path::new("models/test.csv"),
            &Dataset::new(Array2::from(features), Array1::from(real.clone()))
                .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist", "Label"]),
            false,
        ) {
            Ok(_) => (),
            Err(why) => println!("Could not save test features: {}", why),
        }

        (real, predictions, n_packets as f32 / duration)
    }

    pub fn push(&mut self, packet: Packet) -> Option<(Features, bool, (i64, i64))> {
        let mut prediction: Option<(Features, bool, (i64, i64))> = None;
        if self.window.len() < self.window.capacity() {
            self.window.push(packet);
        } else {
            self.window.remove(0);
            self.window.push(packet);
            self.counter += 1;
            if self.counter == self.slide {
                prediction = if let Some(pred) = self.predict() {
                    Some((
                        pred.0,
                        pred.1,
                        (
                            self.window[0].timestamp,
                            self.window[self.window.capacity() - 1].timestamp,
                        ),
                    ))
                } else {
                    None
                };
                self.counter = 0;
            }
        }

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
        let mut hamming: Vec<f64> = Vec::new();

        if let Some(ids) = &self.monitor {
            for p in &self.window {
                if ids.contains(&p.id) {
                    let val = feat.entry(&p.id).or_insert((Vec::new(), Vec::new()));
                    val.0.push(p.timestamp);
                    val.1.push(&p.data);
                }
            }
        } else {
            for p in &self.window {
                let val = feat.entry(&p.id).or_insert((Vec::new(), Vec::new()));
                val.0.push(p.timestamp);
                val.1.push(&p.data);
            }
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
                    // Calculate average Hamming distance
                    hamming.push(
                        val.1
                            .windows(2)
                            .map(|b| {
                                let mut count = 0;
                                for (b1, b2) in b[0].iter().zip(b[1]) {
                                    if *b1 != *b2 {
                                        count += 1
                                    }
                                }
                                count
                            })
                            .collect::<Vec<u32>>()
                            .iter()
                            .sum::<u32>() as f64
                            / (val.1.len()) as f64,
                    );
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
                }
            }

            Some([
                avg_time.iter().sum::<f64>() / avg_time.len() as f64,
                entropy.iter().sum::<f64>() / entropy.len() as f64,
                hamming.iter().sum::<f64>() / hamming.len() as f64,
            ])
        } else {
            None
        }
    }
}

pub mod svm {
    use linfa_svm::Svm;
    use std::fs;
    use std::path::Path;

    pub fn load(path: &Path) -> Result<Svm<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        Ok(bincode::deserialize(&bin).unwrap())
    }

    // #[allow(clippy::type_complexity)]
    // pub fn grid_search(
    //     train: &Dataset<f64, bool>,
    //     test: &Dataset<f64, bool>,
    // ) -> Result<Vec<(String, f64, f64, f64, f64, u8, u8, f32)>, Error> {
    //     struct Parameters {
    //         c: [f64; 4],
    //         eps: [f64; 5],
    //         nu: [f64; 10],
    //         degree: [u8; 4],
    //         cnst: [u8; 5],
    //         kernel: [String; 3],
    //     }
    //     let params = Parameters {
    //         c: [0.1, 1.0, 5.0, 10.0],
    //         eps: [0.0001, 0.01, 0.1, 1.0, 10.0],
    //         nu: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    //         degree: [2, 3, 4, 5],
    //         cnst: [1, 2, 3, 4, 5],
    //         kernel: [
    //             String::from("linear"),
    //             String::from("poly"),
    //             String::from("gaussian"),
    //         ],
    //     };
    //     let mut results = Vec::<(String, f64, f64, f64, f64, u8, u8, f32)>::new();
    //     for kernel in params.kernel {
    //         if kernel == "gaussian" {
    //             for eps in params.eps {
    //                 for c1 in params.c {
    //                     for c2 in params.c {
    //                         for nu in params.nu {
    //                             match Svm::<f64, Pr>::params().pos_neg_weights(c1, c2).gaussian_kernel(eps).nu_weight(nu).fit(train)
    //                             {
    //                                 Ok(model) => match model.predict(test).confusion_matrix(test) {
    //                                     Ok(confusion_matrix) => {
    //                                         println!(
    //                                             "Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {:.3}; {:.3}",
    //                                             kernel,
    //                                             eps,
    //                                             c1,
    //                                             c2,
    //                                             nu,
    //                                             confusion_matrix.f1_score() * 100.0,
    //                                             confusion_matrix.accuracy() * 100.0
    //                                         );
    //                                         results.push((
    //                                             kernel.clone(),
    //                                             eps,
    //                                             c1,
    //                                             c2,
    //                                             nu,
    //                                             0,
    //                                             0,
    //                                             confusion_matrix.f1_score(),
    //                                         ));
    //                                     }
    //                                     Err(why) => {
    //                                         panic!("Could not compute confusion matrix: {}", why)
    //                                     }
    //                                 },
    //                                 Err(why) => panic!("Could not train SVM with parameters - Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {}", kernel, eps, c1, c2, nu, why),
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if kernel == "linear" {
    //             for nu in params.nu {
    //                 for c1 in params.c {
    //                     for c2 in params.c {
    //                         match Svm::<f64, _>::params()
    //                             .pos_neg_weights(c1, c2)
    //                             .nu_weight(nu)
    //                             .fit(train)
    //                         {
    //                             Ok(model) => match model.predict(test).confusion_matrix(test) {
    //                                 Ok(confusion_matrix) => {
    //                                     println!(
    //                                         "Nu: {}; C-pos: {}; C-neg: {} - {:.3}; {:.3}",
    //                                         nu,
    //                                         c1,
    //                                         c2,
    //                                         confusion_matrix.f1_score() * 100.0,
    //                                         confusion_matrix.accuracy() * 100.0
    //                                     );
    //                                     if !(confusion_matrix.f1_score().is_nan()
    //                                         || confusion_matrix.accuracy().is_nan())
    //                                     {
    //                                         results.push((
    //                                             kernel.clone(),
    //                                             -1.0,
    //                                             nu,
    //                                             c1,
    //                                             c2,
    //                                             0,
    //                                             0,
    //                                             confusion_matrix.f1_score(),
    //                                         ));
    //                                     }
    //                                 }
    //                                 Err(why) => {
    //                                     panic!("Could not compute confusion matrix: {}", why)
    //                                 }
    //                             },
    //                             Err(why) => panic!(
    //                                 "Could not train SVM with parameters {}, {}, and {}: {}",
    //                                 nu, c1, c2, why
    //                             ),
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if kernel == "poly" {
    //             for nu in params.nu {
    //                 for c1 in params.c {
    //                     for c2 in params.c {
    //                         for c in params.cnst {
    //                             for d in params.degree {
    //                                 match Svm::<f64, _>::params().polynomial_kernel(c as f64, d as f64)
    //                                     .pos_neg_weights(c1, c2)
    //                                     .nu_weight(nu)
    //                                     .fit(train)
    //                                 {
    //                                     Ok(model) => match model.predict(test).confusion_matrix(test) {
    //                                         Ok(confusion_matrix) => {
    //                                             println!(
    //                                                 "Nu: {}; C-pos: {}; C-neg: {}; Constant: {}; Degree: {} - {:.3}; {:.3}",
    //                                                 nu,
    //                                                 c1,
    //                                                 c2,
    //                                                 c,
    //                                                 d,
    //                                                 confusion_matrix.f1_score() * 100.0,
    //                                                 confusion_matrix.accuracy() * 100.0
    //                                             );
    //                                             if !(confusion_matrix.f1_score().is_nan()
    //                                                 || confusion_matrix.accuracy().is_nan())
    //                                             {
    //                                                 results.push((
    //                                                     kernel.clone(),
    //                                                     -1.0,
    //                                                     nu,
    //                                                     c1,
    //                                                     c2,
    //                                                     c,
    //                                                     d,
    //                                                     confusion_matrix.f1_score(),
    //                                                 ));
    //                                             }
    //                                         }
    //                                         Err(why) => {
    //                                             panic!("Could not compute confusion matrix: {}", why)
    //                                         }
    //                                     },
    //                                     Err(why) => panic!(
    //                                         "Could not train SVM with parameters {}, {}, and {}: {}",
    //                                         nu, c1, c2, why
    //                                     ),
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     results.sort_by(|x, y| y.7.partial_cmp(&x.7).unwrap());
    //     Ok(results)
    // }
}

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
