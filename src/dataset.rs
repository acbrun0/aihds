use crate::model;
use csv::{StringRecord, Writer};
use linfa::dataset::Dataset;
use ndarray_stats::QuantileExt;
use std::{iter::Iterator, path::Path};

// 0 -> True positive
// 1 -> True negative
// 2 -> False positive
// 3 -> False negative
pub fn write_features(path: &Path, dataset: &Dataset<f64, u8>) -> Result<(), csv::Error> {
    let mut wtr = Writer::from_path(path)?;
    match wtr.write_record(dataset.feature_names()) {
        Ok(()) => {
            for (record, target) in dataset.records.outer_iter().zip(dataset.targets.clone()) {
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
    Ok(())
}

pub fn write_features_unsupervised(
    path: &Path,
    dataset: &Dataset<f64, ()>,
) -> Result<(), csv::Error> {
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
    Ok(())
}

pub fn normalize(
    dataset: &mut Dataset<f64, ()>,
    params: &Option<Vec<(f64, f64)>>,
) -> Option<Vec<(f64, f64)>> {
    if dataset.records.is_empty() {
        None
    } else if let Some(params) = params {
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

pub fn packets_from_csv(paths: Vec<&Path>) -> Result<Vec<model::Packet>, csv::Error> {
    let mut packets = Vec::new();
    for path in paths {
        println!("Loading packets from {}", path.display());
        for (i, record) in csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_path(path)?
            .records()
            .enumerate()
        {
            let fields: StringRecord = record?;

            // Converts timestamp from seconds to nanoseconds
            let timestamp = match fields.get(0).unwrap().parse::<f64>() {
                Ok(timestamp) => (timestamp * 1000000_f64) as i64,
                Err(why) => panic!(
                    "Could not parse {} to an integer: {}",
                    fields.get(0).unwrap(),
                    why
                ),
            };

            let id = u32::from_str_radix(fields.get(1).unwrap(), 16).unwrap();

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
                *item = match u8::from_str_radix(fields.get(i + 3).unwrap(), 16) {
                    Ok(digit) => digit,
                    Err(why) => panic!(
                        "Could not parse {} at {:?}: {}",
                        fields.get(i + 3).unwrap(),
                        fields,
                        why
                    ),
                }
            }

            let flag = {
                if let Some(flag) = fields.get((3 + dlc) as usize) {
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
