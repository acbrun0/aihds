use crate::dataset;
use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2, ArrayBase, Dim, OwnedRepr};
use std::{collections::HashMap, fs, path::Path, time::Instant};

pub struct Packet {
    timestamp: Instant,
    id: u32,
    data: Vec<u8>,
}

impl Packet {
    pub fn new(timestamp: Instant, id: u32, data: Vec<u8>) -> Packet {
        Packet {
            timestamp,
            id,
            data,
        }
    }
}

pub struct Ids {
    model: Option<Svm<f64, bool>>,
    scaler: Option<Vec<(f64, f64)>>,
    window: Vec<Packet>,
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
            slide: window_slide,
            counter: 0,
            monitor,
        }
    }

    pub fn train(&mut self, packets: Vec<Packet>) {
        let mut features: Vec<[f64; 3]> = Vec::new();
        let mut labels = Vec::new();

        for packet in packets {
            if self.window.len() < self.window.capacity() {
                self.window.push(packet);
            } else {
                self.window.remove(0);
                self.window.push(packet);
                self.counter += 1;
                if self.counter == self.slide {
                    let mut feat: [f64; 3] = [0.0, 0.0, 0.0];
                    for (i, f) in self.extract_features().iter().enumerate() {
                        feat[i] = *f;
                    }
                    features.push(feat);
                    labels.push(());
                    self.counter = 0;
                }
            }
        }

        let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels));
        self.scaler = dataset::normalize_unsupervised(&mut dataset, &None);

        match svm::train(&dataset) {
            Ok(model) => {
                match std::fs::create_dir_all(Path::new("models")) {
                    Ok(_) => {
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

    pub fn push(&mut self, packet: Packet) -> Option<(Vec<f64>, bool)> {
        let mut prediction: Option<(Vec<f64>, bool)> = None;

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

    fn predict(&self) -> (Vec<f64>, bool) {
        if let Some(scaler) = &self.scaler {
            if let Some(model) = &self.model {
                let mut si = 0;
                let features: Vec<f64> = self
                    .extract_features()
                    .iter()
                    .map(|f| {
                        si += 1;
                        (f - scaler[si - 1].0) / (scaler[si - 1].1 - scaler[si - 1].0)
                    })
                    .collect();
                (features.clone(), !model.predict(Array1::from(features)))
            } else {
                panic!("IDS does not have a model.");
            }
        } else {
            panic!("IDS does not have a scaler.");
        }
    }

    fn extract_features(&self) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
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
                    let stat = feat.entry(p.id).or_insert((Vec::new(), Vec::new()));
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
                let stat = feat.entry(p.id).or_insert((Vec::new(), Vec::new()));
                stat.0.push(p.timestamp);
                stat.1.push(p.data.clone());
            }
        }

        if !feat.is_empty() {
            let mut interval = Vec::new();
            if ts.len() > 1 {
                interval = ts
                    .windows(2)
                    .map(|w| w[1].duration_since(w[0]).as_micros())
                    .collect::<Vec<u128>>();
            }
            if interval.is_empty() {
                interval.push(0);
            }

            for (_, val) in feat.iter_mut() {
                let mut id_interval = Vec::new();
                if val.0.len() > 1 {
                    id_interval = val
                        .0
                        .windows(2)
                        .map(|w| w[1].duration_since(w[0]).as_micros())
                        .collect::<Vec<u128>>();
                    id_interval.swap_remove(0);
                } else {
                    id_interval.push(0);
                }

                avg_time.push(if !val.0.is_empty() {
                    id_interval.iter().sum::<u128>() as f64 / val.0.len() as f64
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

            Array1::from(vec![
                // feat.len() as f64,
                // ts.iter().sum::<f64>() / ts.len() as f64,
                // general_entropy,
                avg_time.iter().sum::<f64>() / avg_time.len() as f64,
                entropy.iter().sum::<f64>() / entropy.len() as f64,
                hamming.iter().sum::<f64>() / hamming.len() as f64,
            ])
        } else {
            Array1::from(vec![0.0, 0.0, 0.0])
        }
    }
}

pub mod svm {
    use linfa::prelude::*;
    use linfa_svm::Svm;
    use ndarray::{ArrayBase, Dim, ViewRepr};
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    pub fn train(dataset: &Dataset<f64, ()>) -> Result<Svm<f64, bool>, linfa_svm::SvmError> {
        Svm::<f64, _>::params()
            .gaussian_kernel(0.01)
            // .polynomial_kernel(2.0, 3.0)
            .nu_weight(0.01)
            .fit(dataset)
    }

    pub fn predict(
        model: &Svm<f64, bool>,
        features: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>,
    ) -> bool {
        model.predict(features)
    }

    pub fn test(
        dataset: &Dataset<f64, bool>,
        model: Svm<f64, bool>,
    ) -> (Result<ConfusionMatrix<bool>, linfa::Error>, f64) {
        let start = Instant::now();
        let pred = model.predict(dataset);
        (
            // pred.confusion_matrix(dataset),
            pred.mapv(|p| !p).confusion_matrix(dataset),
            dataset.records.shape()[0] as f64 / start.elapsed().as_secs_f64(),
        )
    }

    pub fn load(path: &Path) -> Result<Svm<f64, bool>, std::io::Error> {
        let bin = fs::read(path)?;
        let model: Svm<f64, bool> = bincode::deserialize(&bin).unwrap();
        Ok(model)
    }

    #[allow(clippy::type_complexity)]
    pub fn grid_search(
        train: &Dataset<f64, ()>,
        test: &Dataset<f64, bool>,
    ) -> Result<Vec<(String, f64, f64, f64, f64, u8, u8, f32)>, Error> {
        struct Parameters {
            c: [f64; 4],
            eps: [f64; 5],
            nu: [f64; 10],
            degree: [u8; 4],
            cnst: [u8; 5],
            kernel: [String; 3],
        }
        let params = Parameters {
            c: [0.1, 1.0, 5.0, 10.0],
            eps: [0.0001, 0.01, 0.1, 1.0, 10.0],
            nu: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            degree: [2, 3, 4, 5],
            cnst: [1, 2, 3, 4, 5],
            kernel: [
                String::from("linear"),
                String::from("poly"),
                String::from("gaussian"),
            ],
        };
        let mut results = Vec::<(String, f64, f64, f64, f64, u8, u8, f32)>::new();
        for kernel in params.kernel {
            if kernel == "gaussian" {
                for eps in params.eps {
                    for c1 in params.c {
                        for c2 in params.c {
                            for nu in params.nu {
                                match Svm::<f64, _>::params().pos_neg_weights(c1, c2).gaussian_kernel(eps).nu_weight(nu).fit(train)
                                {
                                    Ok(model) => match model.predict(test).confusion_matrix(test) {
                                        Ok(confusion_matrix) => {
                                            println!(
                                                "Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {:.3}; {:.3}",
                                                kernel,
                                                eps,
                                                c1,
                                                c2,
                                                nu,
                                                confusion_matrix.f1_score() * 100.0,
                                                confusion_matrix.accuracy() * 100.0
                                            );
                                            results.push((
                                                kernel.clone(),
                                                eps,
                                                c1,
                                                c2,
                                                nu,
                                                0,
                                                0,
                                                confusion_matrix.f1_score(),
                                            ));
                                        }
                                        Err(why) => {
                                            panic!("Could not compute confusion matrix: {}", why)
                                        }
                                    },
                                    Err(why) => panic!("Could not train SVM with parameters - Kernel: {}; Eps: {}; C-Pos: {}, C-Neg: {}, Nu: {} - {}", kernel, eps, c1, c2, nu, why),
                                }
                            }
                        }
                    }
                }
            } else if kernel == "linear" {
                for nu in params.nu {
                    for c1 in params.c {
                        for c2 in params.c {
                            match Svm::<f64, _>::params()
                                .pos_neg_weights(c1, c2)
                                .nu_weight(nu)
                                .fit(train)
                            {
                                Ok(model) => match model.predict(test).confusion_matrix(test) {
                                    Ok(confusion_matrix) => {
                                        println!(
                                            "Nu: {}; C-pos: {}; C-neg: {} - {:.3}; {:.3}",
                                            nu,
                                            c1,
                                            c2,
                                            confusion_matrix.f1_score() * 100.0,
                                            confusion_matrix.accuracy() * 100.0
                                        );
                                        if !(confusion_matrix.f1_score().is_nan()
                                            || confusion_matrix.accuracy().is_nan())
                                        {
                                            results.push((
                                                kernel.clone(),
                                                -1.0,
                                                nu,
                                                c1,
                                                c2,
                                                0,
                                                0,
                                                confusion_matrix.f1_score(),
                                            ));
                                        }
                                    }
                                    Err(why) => {
                                        panic!("Could not compute confusion matrix: {}", why)
                                    }
                                },
                                Err(why) => panic!(
                                    "Could not train SVM with parameters {}, {}, and {}: {}",
                                    nu, c1, c2, why
                                ),
                            }
                        }
                    }
                }
            } else if kernel == "poly" {
                for nu in params.nu {
                    for c1 in params.c {
                        for c2 in params.c {
                            for c in params.cnst {
                                for d in params.degree {
                                    match Svm::<f64, _>::params().polynomial_kernel(c as f64, d as f64)
                                        .pos_neg_weights(c1, c2)
                                        .nu_weight(nu)
                                        .fit(train)
                                    {
                                        Ok(model) => match model.predict(test).confusion_matrix(test) {
                                            Ok(confusion_matrix) => {
                                                println!(
                                                    "Nu: {}; C-pos: {}; C-neg: {}; Constant: {}; Degree: {} - {:.3}; {:.3}",
                                                    nu,
                                                    c1,
                                                    c2,
                                                    c,
                                                    d,
                                                    confusion_matrix.f1_score() * 100.0,
                                                    confusion_matrix.accuracy() * 100.0
                                                );
                                                if !(confusion_matrix.f1_score().is_nan()
                                                    || confusion_matrix.accuracy().is_nan())
                                                {
                                                    results.push((
                                                        kernel.clone(),
                                                        -1.0,
                                                        nu,
                                                        c1,
                                                        c2,
                                                        c,
                                                        d,
                                                        confusion_matrix.f1_score(),
                                                    ));
                                                }
                                            }
                                            Err(why) => {
                                                panic!("Could not compute confusion matrix: {}", why)
                                            }
                                        },
                                        Err(why) => panic!(
                                            "Could not train SVM with parameters {}, {}, and {}: {}",
                                            nu, c1, c2, why
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        results.sort_by(|x, y| y.7.partial_cmp(&x.7).unwrap());
        Ok(results)
    }
}

pub fn save<S: serde::ser::Serialize>(model: &S, path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bincode::serialize(model).unwrap())?;
    Ok(())
}
