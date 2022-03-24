use crate::dataset;
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use std::{collections::HashMap, fs, io::Write, path::Path};

pub type Features = [f64; 3];

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
                if let Some(monitor) = &self.monitor {
                    if monitor.contains(&packet.id) {
                        self.window.push(packet)
                    }
                } else {
                    self.window.push(packet);
                }
            } else if let Some(monitor) = &self.monitor {
                if monitor.contains(&packet.id) {
                    self.window.remove(0);
                    self.window.push(packet);
                    self.counter += 1;
                    if self.counter == self.slide {
                        let mut feat: Features = [0.0, 0.0, 0.0];
                        for (i, f) in self.extract_features().iter().enumerate() {
                            feat[i] = *f;
                        }
                        features.push(feat);
                        labels.push(());
                        self.counter = 0;
                    }
                }
            } else {
                self.window.remove(0);
                self.window.push(packet);
                self.counter += 1;
                if self.counter == self.slide {
                    let mut feat: Features = [0.0, 0.0, 0.0];
                    for (i, f) in self.extract_features().iter().enumerate() {
                        feat[i] = *f;
                    }
                    features.push(feat);
                    labels.push(());
                    self.counter = 0;
                }
            }
        }

        let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
            .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist"]);
        self.scaler = dataset::normalize_unsupervised(&mut dataset, &None);

        match Svm::<f64, _>::params()
            .gaussian_kernel(5.0)
            .nu_weight(0.001)
            .fit(&dataset)
        {
            Ok(model) => {
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

    pub fn test(&mut self, packets: Vec<Packet>) -> (Vec<bool>, Vec<(Features, bool)>) {
        let mut predictions = Vec::new();
        let mut real = Vec::new();
        let mut features = Vec::new();
        match fs::File::create(Path::new("models/test.txt")) {
            Ok(mut file) => {
                for packet in packets {
                    if let Some(result) = self.push(packet) {
                        match file.write_all(format!("{:?} -> {}\n", result.0, result.1).as_bytes())
                        {
                            Ok(_) => (),
                            Err(why) => panic!("Could not write log: {}", why),
                        }
                        features.push(result.0);
                        predictions.push(result);
                        real.push(self.window.iter().any(|p| p.flag));
                    }
                }
            }
            Err(why) => panic!("Could not create log file: {}", why),
        }

        match dataset::write_features(
            Path::new("models/test.csv"),
            &Dataset::new(Array2::from(features), Array1::from(real.clone()))
                .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist", "Label"]),
            false,
        ) {
            Ok(_) => (),
            Err(why) => println!("Could not save test features: {}", why),
        }

        (real, predictions)
    }

    pub fn push(&mut self, packet: Packet) -> Option<(Features, bool)> {
        let mut prediction: Option<(Features, bool)> = None;

        if self.window.len() < self.window.capacity() {
            self.window.push(packet);
        } else {
            self.window.remove(0);
            self.window.push(packet);
            self.counter += 1;
            if self.counter == self.slide {
                prediction = Some(self.predict());
                self.counter = 0;
            }
        }

        prediction
    }

    fn predict(&self) -> (Features, bool) {
        if let Some(scaler) = &self.scaler {
            if let Some(model) = &self.model {
                let mut si = 0;
                let features: Features = self
                    .extract_features()
                    .iter()
                    .map(|f| {
                        si += 1;
                        (f - scaler[si - 1].0) / (scaler[si - 1].1 - scaler[si - 1].0)
                    })
                    .collect::<Vec<f64>>()
                    .as_slice()
                    .try_into()
                    .unwrap();
                (features, model.predict(Array1::from(features.to_vec())))
            } else {
                panic!("IDS does not have a model.");
            }
        } else {
            panic!("IDS does not have a scaler.");
        }
    }

    fn extract_features(&self) -> Features {
        let mut feat = HashMap::new();
        let mut ts = Vec::new();
        let mut avg_time = Vec::new();
        let mut entropy = Vec::new();
        let mut hamming: Vec<f64> = Vec::new();
        let mut _general_entropy = 0.0;

        if let Some(ids) = &self.monitor {
            for p in &self.window {
                if ids.contains(&p.id) {
                    ts.push(p.timestamp);
                    let prob = self.window.iter().filter(|&x| x.data == p.data).count() as f64
                        / self.window.len() as f64;
                    _general_entropy += 0.0 - prob * prob.log2();
                    let stat = feat.entry(&p.id).or_insert((Vec::new(), Vec::new()));
                    stat.0.push(p.timestamp);
                    stat.1.push(p.data.clone());
                }
            }
        } else {
            for p in &self.window {
                ts.push(p.timestamp);
                let prob = self.window.iter().filter(|&x| x.data == p.data).count() as f64
                    / self.window.len() as f64;
                _general_entropy += 0.0 - prob * prob.log2();
                let stat = feat.entry(&p.id).or_insert((Vec::new(), Vec::new()));
                stat.0.push(p.timestamp);
                stat.1.push(p.data.clone());
            }
        }

        if !feat.is_empty() {
            let mut interval = Vec::new();
            if ts.len() > 1 {
                interval = ts.windows(2).map(|w| w[1] - w[0]).collect::<Vec<i64>>();
            }
            if interval.is_empty() {
                interval.push(0);
            }

            for (_, val) in feat.iter_mut() {
                let mut id_interval = Vec::new();
                if val.0.len() > 1 {
                    id_interval = val.0.windows(2).map(|w| w[1] - w[0]).collect::<Vec<i64>>();
                } else {
                    id_interval.push(0);
                }

                avg_time.push(if !val.0.is_empty() {
                    id_interval.iter().sum::<i64>() as f64 / val.0.len() as f64
                } else {
                    0.0
                });

                let n_packets = val.1.len();
                let mut datamap = HashMap::new();
                let mut probs = Vec::new();
                for bytes in &val.1 {
                    let entry = datamap.entry(bytes).or_insert(0);
                    *entry += 1;
                }
                for count in datamap.values() {
                    probs.push(*count as f64 / n_packets as f64);
                }
                entropy.push(0.0 - probs.iter().map(|p| p * p.log2()).sum::<f64>());
                if val.1.len() > 1 {
                    hamming.push(
                        val.1
                            .windows(2)
                            .map(|b| {
                                let mut count = 0;
                                for (b1, b2) in b[0].iter().zip(b[1].clone()) {
                                    if *b1 != b2 {
                                        count += 1
                                    }
                                }
                                count
                            })
                            .collect::<Vec<u32>>()
                            .iter()
                            .sum::<u32>() as f64
                            / (val.1.len() - 1) as f64,
                    );
                }
            }

            [
                // feat.len() as f64,
                // ts.iter().sum::<f64>() / ts.len() as f64,
                // general_entropy,
                avg_time.iter().sum::<f64>() / avg_time.len() as f64,
                entropy.iter().sum::<f64>() / entropy.len() as f64,
                hamming.iter().sum::<f64>() / hamming.len() as f64,
            ]
        } else {
            [0.0, 0.0, 0.0]
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
