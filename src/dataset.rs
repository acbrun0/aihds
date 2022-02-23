use csv::{StringRecord, Writer};
use linfa::dataset::Dataset;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use std::fs;
use std::io::prelude::*;
use std::iter::Iterator;
use std::path::Path;

struct Packet {
    timestamp: f64,
    id: u16,
    data: [u8; 8],
    flag: bool,
}

const WINDOW_SIZE: usize = 250;
const WINDOW_SLIDE: usize = 50;

pub fn write_features(
    path: &Path,
    dataset: &mut Dataset<f64, bool>,
    is_libsvm: bool,
) -> Result<(), csv::Error> {
    let scaler = normalize(dataset, None);
    std::fs::write(path, bincode::serialize(&scaler).unwrap())
        .expect("Could not serialize scaler to file.");
    if is_libsvm {
        let mut file = fs::File::create(path).expect("Could not create file.");
        let targets = dataset.targets.clone().into_raw_vec();
        for (i, record) in dataset.records.outer_iter().enumerate() {
            file.write_all(
                format!(
                    "{} 1:{} 2:{} 3:{} 4:{}\n",
                    if targets[i] { "-1" } else { "+1" },
                    record[0],
                    record[1],
                    record[2],
                    record[3]
                )
                .as_bytes(),
            )
            .expect("Unable to write to file.");
        }
    } else {
        let mut wtr = Writer::from_path(path)?;
        for (i, record) in dataset.records.outer_iter().enumerate() {
            wtr.write_record(&[
                record[0].to_string(),
                record[1].to_string(),
                record[2].to_string(),
                record[3].to_string(),
                dataset.targets.slice(s![i, 0]).to_string(),
            ])?;
        }
        wtr.flush()?;
    }
    Ok(())
}

fn normalize(
    dataset: &mut Dataset<f64, bool>,
    params: Option<Vec<(f64, f64)>>,
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

fn normalize_unsupervised(
    dataset: &mut Dataset<f64, ()>,
    params: Option<Vec<(f64, f64)>>,
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

fn standardize(
    dataset: &mut Dataset<f64, bool>,
    params: Option<Vec<(f64, f64)>>,
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

fn standardize_unsupervised(
    dataset: &mut Dataset<f64, ()>,
    params: Option<Vec<(f64, f64)>>,
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

// Extracts:
//  - Number of distinct IDs;
//  - Average time between packets;
//  - Average time between packets of the same ID;
//  - General entropy
//  - Entropy between packets of the same ID;
fn extract_features(packets: &[Packet]) -> [f64; 5] {
    let mut feat = HashMap::new();
    let mut ts = Vec::new();
    let mut avg_time = Vec::new();
    let mut entropy = Vec::new();
    let mut general_entropy = 0.0;

    for p in packets {
        ts.push(p.timestamp);
        let prob =
            packets.iter().filter(|&x| x.data == p.data).count() as f64 / packets.len() as f64;
        general_entropy += 0.0 - prob * prob.log2();
        let stat = feat.entry(p.id).or_insert((Vec::new(), Vec::new()));
        stat.0.push(p.timestamp);
        stat.1.push(p.data);
    }

    ts = ts.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();
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
        // let mut ham = Vec::new();
        for bytes in &val.1 {
            let entry = datamap.entry(bytes).or_insert(0);
            *entry += 1;
        }
        for count in datamap.values() {
            probs.push(*count as f64 / n_packets as f64);
        }
        entropy.push(0.0 - probs.iter().map(|p| p * p.log2()).sum::<f64>());
    }

    [
        feat.len() as f64,
        ts.iter().sum::<f64>() / ts.len() as f64,
        avg_time.iter().sum::<f64>() / avg_time.len() as f64,
        general_entropy,
        entropy.iter().sum::<f64>() / entropy.len() as f64,
    ]
}

pub fn load_unsupervised(
    paths: Vec<&Path>,
    scaler: Option<Vec<(f64, f64)>>,
    scale: bool,
) -> Result<Dataset<f64, ()>, csv::Error> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for path in &paths {
        let mut window: Vec<Packet> = Vec::with_capacity(WINDOW_SIZE);
        let mut buffer: Vec<Packet> = Vec::with_capacity(WINDOW_SLIDE);
        println!("Loading {}", path.display());
        for record in csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)?
            .records()
        {
            let fields: StringRecord = record?;

            let timestamp = match fields.get(0).unwrap().parse() {
                Ok(t) => t,
                Err(why) => panic!("Could not parse: {}", why),
            };

            let id = u16::from_str_radix(fields.get(1).unwrap(), 16).unwrap();

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
                    window.drain(..WINDOW_SLIDE);
                }
                window.append(&mut buffer);
                features.push(extract_features(&window));
                let mut flag = false;
                for p in &window {
                    if p.flag {
                        flag = true;
                        break;
                    }
                }
                labels.push(flag);
            }
            buffer.push(Packet {
                timestamp,
                id,
                data,
                flag,
            });
        }
    }
    let mut empty = Vec::new();
    for _ in 0..features.len() {
        empty.push(());
    }
    let mut dataset =
        Dataset::new(Array2::from(features), Array1::from(empty)).with_feature_names(vec![
            "Distinct IDs",
            "Average time between consecutive packets",
            "Average time between packets of the same ID",
            "Average entropy between packets of the same ID",
        ]);
    if scale {
        if let Some(scaler) = scaler {
            normalize_unsupervised(&mut dataset, Some(scaler));
            // standardize_unsupervised(&mut dataset, Some(scaler));
        } else {
            let scaler = normalize_unsupervised(&mut dataset, None);
            // let scaler = standardize_unsupervised(&mut dataset, None);
            std::fs::write(
                "datasets/scaler",
                bincode::serialize(&scaler).expect("Could not serialize scaler."),
            )
            .expect("Could not save scaler to file.");
        }
    }
    Ok(dataset)
}

pub fn load(
    paths: Vec<&Path>,
    scaler: Option<Vec<(f64, f64)>>,
    scale: bool,
) -> Result<Dataset<f64, bool>, csv::Error> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for path in &paths {
        let mut window: Vec<Packet> = Vec::with_capacity(WINDOW_SIZE);
        let mut buffer: Vec<Packet> = Vec::with_capacity(WINDOW_SLIDE);
        println!("Loading {}", path.display());
        for record in csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)?
            .records()
        {
            let fields: StringRecord = record?;

            let timestamp = match fields.get(0).unwrap().parse() {
                Ok(t) => t,
                Err(why) => panic!("Could not parse: {}", why),
            };

            let id = u16::from_str_radix(fields.get(1).unwrap(), 16).unwrap();

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
                    window.drain(..WINDOW_SLIDE);
                }
                window.append(&mut buffer);
                features.push(extract_features(&window));
                let mut flag = false;
                for p in &window {
                    if p.flag {
                        flag = true;
                        break;
                    }
                }
                labels.push(flag);
            }
            buffer.push(Packet {
                timestamp,
                id,
                data,
                flag,
            });
        }
    }
    let mut dataset = Dataset::new(Array2::from(features), Array1::from(labels))
        .with_feature_names(vec![
            "Distinct IDs",
            "Average time between consecutive packets",
            "Average time between packets of the same ID",
            "Average entropy between packets of the same ID",
        ]);
    if scale {
        if let Some(scaler) = scaler {
            // normalize(&mut dataset, Some(scaler));
            standardize(&mut dataset, Some(scaler));
        } else {
            // let scaler = normalize(&mut dataset, None);
            let scaler = standardize(&mut dataset, None);
            std::fs::write(
                "datasets/scaler",
                bincode::serialize(&scaler).expect("Could not serialize scaler."),
            )
            .expect("Could not save scaler to file.");
        }
    }
    Ok(dataset)
}
