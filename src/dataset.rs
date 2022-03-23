use crate::model;
use csv::{StringRecord, Writer};
use linfa::dataset::Dataset;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::{collections::HashMap, fs, io::prelude::*, iter::Iterator, path::Path};

struct Packet {
    timestamp: f64,
    id: String,
    data: [u8; 8],
    flag: bool,
}

const WINDOW_SIZE: usize = 200;
const WINDOW_SLIDE: usize = 50;
const ATTACK_THRESHOLD: f64 = 0.1;

#[allow(dead_code)]
pub fn write_features(
    path: &Path,
    dataset: &Dataset<f64, bool>,
    is_libsvm: bool,
) -> Result<(), csv::Error> {
    if is_libsvm {
        let mut file = fs::File::create(path).expect("Could not create file.");
        for (record, target) in dataset
            .records
            .outer_iter()
            .zip(dataset.targets.clone().into_raw_vec())
        {
            file.write_all(
                format!(
                    "{} 1:{} 2:{} 3:{}\n",
                    if target { "-1" } else { "+1" },
                    record[0],
                    record[1],
                    record[2]
                )
                .as_bytes(),
            )
            .expect("Unable to write to file.");
        }
    } else {
        let mut wtr = Writer::from_path(path)?;
        match wtr.write_record(dataset.feature_names()) {
            Ok(()) => {
                for (record, target) in dataset
                    .records
                    .outer_iter()
                    .zip(dataset.targets.clone())
                {
                    wtr.write_record(&[
                        record[0].to_string(),
                        record[1].to_string(),
                        record[2].to_string(),
                        target.to_string(),
                    ])?;
                }
                wtr.flush()?;
            }
            Err(why) => panic!("Could not write to {}: {}", path.display(), why),
        }
    }
    Ok(())
}

pub fn write_features_unsupervised(
    path: &Path,
    dataset: &Dataset<f64, ()>,
    is_libsvm: bool,
) -> Result<(), csv::Error> {
    if is_libsvm {
        let mut file = fs::File::create(path).expect("Could not create file.");
        let _targets = dataset.targets.clone().into_raw_vec();
        for (_i, record) in dataset.records.outer_iter().enumerate() {
            file.write_all(
                format!("0 1:{} 2:{} 3:{}\n", record[0], record[1], record[2]).as_bytes(),
            )
            .expect("Unable to write to file.");
        }
    } else {
        let mut wtr = Writer::from_path(path)?;
        match wtr.write_record(dataset.feature_names()) {
            Ok(()) => {
                for record in dataset.records.outer_iter() {
                    wtr.write_record(&[
                        record[0].to_string(),
                        record[1].to_string(),
                        record[2].to_string(),
                    ])?;
                }
                wtr.flush()?;
            }
            Err(why) => panic!("Could not write to {}: {}", path.display(), why),
        }
    }
    Ok(())
}

// Extracts:
//  - Number of distinct IDs;
//  - Average time between packets;
//  - Average time between packets of the same ID;
//  - General entropy
//  - Entropy between packets of the same ID;
//  - Average Hamming distance between packets of the same ID
fn extract_features(packets: &[Packet], monitor: &Option<Vec<String>>) -> [f64; 3] {
    let mut feat = HashMap::new();
    let mut ts = Vec::new();
    let mut avg_time = Vec::new();
    let mut entropy = Vec::new();
    let mut hamming: Vec<f64> = Vec::new();
    let mut _general_entropy = 0.0;

    if let Some(ids) = monitor {
        for p in packets {
            if ids.contains(&p.id) {
                ts.push(p.timestamp);
                let prob = packets.iter().filter(|&x| x.data == p.data).count() as f64
                    / packets.len() as f64;
                _general_entropy += 0.0 - prob * prob.log2();
                let stat = feat.entry(p.id.clone()).or_insert((Vec::new(), Vec::new()));
                stat.0.push(p.timestamp);
                stat.1.push(p.data);
            }
        }
    } else {
        for p in packets {
            ts.push(p.timestamp);
            let prob =
                packets.iter().filter(|&x| x.data == p.data).count() as f64 / packets.len() as f64;
            _general_entropy += 0.0 - prob * prob.log2();
            let stat = feat.entry(p.id.clone()).or_insert((Vec::new(), Vec::new()));
            stat.0.push(p.timestamp);
            stat.1.push(p.data);
        }
    }

    if !feat.is_empty() {
        if ts.len() > 1 {
            ts = ts.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();
        }
        ts[0] = 0.0;

        for (_, val) in feat.iter_mut() {
            if val.0.len() > 1 {
                val.0 = val.0.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();
                val.0.swap_remove(0);
            } else {
                val.0[0] = 0.0;
            }

            avg_time.push(if !val.0.is_empty() {
                val.0.iter().sum::<f64>() / val.0.len() as f64
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
                            for (b1, b2) in b[0].iter().zip(b[1]) {
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

#[allow(dead_code)]
fn standardize(
    dataset: &mut Dataset<f64, bool>,
    params: &Option<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
    if let Some(params) = params {
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / params[i].1);
        }
        None
    } else {
        let mut params = Vec::new();
        for col in dataset.records.columns() {
            let count = col.len() as f64;
            let mean = col.iter().sum::<f64>() / count;
            let variance = col
                .iter()
                .map(|v| {
                    let diff = mean - v;
                    diff * diff
                })
                .sum::<f64>()
                / count;
            params.push((mean, variance.sqrt()));
        }
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / params[i].1);
        }
        Some(params)
    }
}

#[allow(dead_code)]
fn standardize_unsupervised(
    dataset: &mut Dataset<f64, ()>,
    params: &Option<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
    if let Some(params) = params {
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / params[i].1);
        }
        None
    } else {
        let mut params = Vec::new();
        for col in dataset.records.columns() {
            let count = col.len() as f64;
            let mean = col.iter().sum::<f64>() / count;
            let variance = col
                .iter()
                .map(|v| {
                    let diff = mean - v;
                    diff * diff
                })
                .sum::<f64>()
                / count;
            params.push((mean, variance.sqrt()));
        }
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / params[i].1);
        }
        Some(params)
    }
}

pub fn normalize(
    dataset: &mut Dataset<f64, bool>,
    params: &Option<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
    if let Some(params) = params {
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / (params[i].1 - params[i].0));
        }
        None
    } else {
        let mut params = Vec::new();
        for col in dataset.records.columns() {
            params.push((*col.min().unwrap(), *col.max().unwrap()));
        }
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / (params[i].1 - params[i].0));
        }
        Some(params)
    }
}

pub fn normalize_unsupervised(
    dataset: &mut Dataset<f64, ()>,
    params: &Option<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
    if let Some(params) = params {
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / (params[i].1 - params[i].0));
        }
        None
    } else {
        let mut params = Vec::new();
        for col in dataset.records.columns() {
            params.push((*col.min().unwrap(), *col.max().unwrap()));
        }
        for (i, mut col) in dataset.records.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|v| (v - params[i].0) / (params[i].1 - params[i].0));
        }
        Some(params)
    }
}

#[allow(clippy::type_complexity)]
pub fn load(
    paths: Vec<&Path>,
    scaler: Option<Vec<(f64, f64)>>,
    monitor: &Option<Vec<String>>,
) -> Result<(Dataset<f64, bool>, Vec<(f64, f64)>), csv::Error> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for path in paths {
        let mut window: Vec<Packet> = Vec::with_capacity(WINDOW_SIZE);
        let mut buffer: Vec<Packet> = Vec::with_capacity(WINDOW_SLIDE);
        println!("Loading {}", path.display());
        for record in csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?
            .records()
        {
            let fields: StringRecord = record?;

            let timestamp = match fields.get(0).unwrap().parse() {
                Ok(t) => t,
                Err(why) => panic!("Could not parse: {}", why),
            };

            let id = fields.get(1).unwrap();

            let dlc: u8 = match fields.get(2).unwrap().parse() {
                Ok(dlc) => dlc,
                Err(why) => panic!("Could not parse: {}", why),
            };

            let mut data = [0; 8];
            for (i, item) in data.iter_mut().enumerate().take(dlc as usize) {
                *item = u8::from_str_radix(fields.get(i + 3).unwrap(), 16).unwrap();
            }

            let flag = if let Some(f) = fields.get(fields.len() - 1) {
                f != "Normal"
            } else {
                false
            };

            if buffer.len() == buffer.capacity() {
                if window.len() == window.capacity() {
                    features.push(extract_features(&window, monitor));
                    // let mut flag = false;
                    // for p in &window {
                    //     if p.flag {
                    //         flag = true;
                    //         break;
                    //     }
                    // }
                    labels.push(
                        window.iter().filter(|&p| p.flag).count() as f64
                            > WINDOW_SIZE as f64 * ATTACK_THRESHOLD,
                    );
                    window.drain(..WINDOW_SLIDE);
                }
                window.append(&mut buffer);
            }
            if window.len() < window.capacity() {
                window.push(Packet {
                    timestamp,
                    id: String::from(id),
                    data,
                    flag,
                });
            } else {
                buffer.push(Packet {
                    timestamp,
                    id: String::from(id),
                    data,
                    flag,
                });
            }
        }
    }
    let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
        .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist", "Label"]);
    if let Some(new_scaler) = normalize(&mut dataset, &scaler) {
        Ok((dataset, new_scaler))
    } else {
        Ok((dataset, scaler.unwrap()))
    }
}

#[allow(clippy::type_complexity)]
pub fn load_unsupervised(
    paths: Vec<&Path>,
    scaler: Option<Vec<(f64, f64)>>,
    monitor: &Option<Vec<String>>,
) -> Result<(Dataset<f64, ()>, Vec<(f64, f64)>), csv::Error> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for path in paths {
        let mut window: Vec<Packet> = Vec::with_capacity(WINDOW_SIZE);
        let mut buffer: Vec<Packet> = Vec::with_capacity(WINDOW_SLIDE);
        println!("Loading {}", path.display());
        for (i, record) in csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?
            .records()
            .enumerate()
        {
            let fields: StringRecord = record?;

            let timestamp = match fields.get(0).unwrap().parse() {
                Ok(t) => t,
                Err(why) => panic!("Could not parse: {}", why),
            };

            let id = fields.get(1).unwrap();

            let dlc: u8 = match fields.get(2).unwrap().parse() {
                Ok(dlc) => dlc,
                Err(why) => panic!(
                    "Could not parse {:?} from record #{}: {}",
                    fields.get(2),
                    i,
                    why
                ),
            };

            let mut data = [0; 8];
            for (i, item) in data.iter_mut().enumerate().take(dlc as usize) {
                *item = u8::from_str_radix(fields.get(i + 3).unwrap(), 16).unwrap();
            }

            let flag = if let Some(f) = fields.get(fields.len() - 1) {
                f != "Normal"
            } else {
                false
            };

            if buffer.len() == buffer.capacity() {
                if window.len() == window.capacity() {
                    features.push(extract_features(&window, monitor));
                    window.drain(..WINDOW_SLIDE);
                }
                window.append(&mut buffer);
                labels.push(());
            }
            if window.len() < window.capacity() {
                window.push(Packet {
                    timestamp,
                    id: String::from(id),
                    data,
                    flag,
                });
            } else {
                buffer.push(Packet {
                    timestamp,
                    id: String::from(id),
                    data,
                    flag,
                });
            }
        }
    }
    let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
        .with_feature_names(vec!["AvgTime", "Entropy", "HammingDist"]);
    if let Some(new_scaler) = normalize_unsupervised(&mut dataset, &scaler) {
        match std::fs::create_dir_all(Path::new("models")) {
            Ok(_) => (),
            Err(why) => panic!("Could not create models directory: {}", why),
        };
        match fs::write("models/scaler", bincode::serialize(&new_scaler).unwrap()) {
            Ok(()) => (),
            Err(why) => println!("Could not save scaler: {}", why),
        }
        Ok((dataset, new_scaler))
    } else {
        Ok((dataset, scaler.unwrap()))
    }
}

pub fn packets_from_csv(paths: Vec<&Path>) -> Result<Vec<model::Packet>, csv::Error> {
    let mut packets = Vec::new();
    for path in paths {
        for (i, record) in csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?
            .records()
            .enumerate()
        {
            let fields: StringRecord = record?;

            // Converts timestamp from seconds to nanoseconds
            let timestamp = match fields.get(0).unwrap().replace('.', "").parse::<i64>(){
                Ok(timestamp) => timestamp * 1000,
                Err(why) => panic!("Could not parse {} to an integer: {}", fields.get(0).unwrap(), why)
            };

            let id = match u32::from_str_radix(fields.get(1).unwrap(), 16) {
                Ok(int) => int,
                Err(why) => panic!(
                    "Could not convert {} to integer: {}",
                    fields.get(1).unwrap(),
                    why
                ),
            };

            let dlc: u8 = match fields.get(2).unwrap().parse() {
                Ok(dlc) => dlc,
                Err(why) => panic!(
                    "Could not parse {:?} from record #{}: {}",
                    fields.get(2),
                    i,
                    why
                ),
            };

            let mut bytes = [0; 8];
            for (i, item) in bytes.iter_mut().enumerate().take(dlc as usize) {
                *item = u8::from_str_radix(fields.get(i + 3).unwrap(), 16).unwrap();
            }

            let flag = {
                if let Some(flag) = fields.get((3 + dlc + 1) as usize) {
                    flag != "Normal"
                } else {
                    false
                }
            };

            packets.push(model::Packet::new(timestamp, id, bytes.to_vec(), flag));
        }
    }
    Ok(packets)
}
