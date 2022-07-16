#![warn(missing_docs)]

//! This module provides functionality around CAN traffic files and structured datasets.

use crate::ids;
use csv::{StringRecord, Writer};
use linfa::dataset::Dataset;
use ndarray::Ix1;
use ndarray_stats::QuantileExt;
use std::{iter::Iterator, path::Path};

/// Writes a labeled dataset to a CSV file.
/// # Examples
/// ```
/// use linfa::dataset::Dataset;
/// use ndarray::{Array1, Array2};
/// use std::path::Path;
///
/// let ds = Dataset::new(Array2::from(vec![[1,2][3,4]]), Array1::from([0,1]));
/// write_features(Path::new("features.csv"), ds);
/// ```
pub fn write_features(path: &Path, dataset: &Dataset<f64, u8, Ix1>) -> Result<(), csv::Error> {
    // 0 -> True positive
    // 1 -> True negative
    // 2 -> False positive
    // 3 -> False negative
    let mut wtr = Writer::from_path(path)?;
    match wtr.write_record(dataset.feature_names()) {
        Ok(()) => {
            for (record, target) in dataset.records.outer_iter().zip(dataset.targets.clone()) {
                let mut line: Vec<String> = record.iter().map(|r| r.to_string()).collect();
                line.push(target.to_string());
                wtr.write_record(line)?;
            }
            wtr.flush()?;
        }
        Err(why) => panic!("Could not write to {}: {}", path.display(), why),
    }
    Ok(())
}

/// Writes a unlabeled dataset to a CSV file.
/// # Examples
/// ```
/// use linfa::dataset::Dataset;
/// use ndarray::{Array1, Array2};
/// use std::path::Path;
///
/// let ds = Dataset::new(Array2::from(vec![[1,2][3,4]]), Array1::from(vec![(), ()]));
/// write_features_unsupervised(Path::new("features.csv"), ds);
/// ```
pub fn write_features_unsupervised(
    path: &Path,
    dataset: &Dataset<f64, (), Ix1>,
) -> Result<(), csv::Error> {
    let mut wtr = Writer::from_path(path)?;
    match wtr.write_record(dataset.feature_names()) {
        Ok(()) => {
            for record in dataset.records.outer_iter() {
                wtr.write_record(record.iter().map(|r| r.to_string()))?;
            }
            wtr.flush()?;
        }
        Err(why) => panic!("Could not write to {}: {}", path.display(), why),
    }
    Ok(())
}

/// Normalize a dataset according to a provided list of minimum and maximum values. If no values are provided, minimum and maximum values are calculated, the dataset is normalized, and the values are returned.
///
/// # Examples
/// ```
/// use linfa::dataset::Dataset;
/// use ndarray::{Array1, Array2};
///
/// let ds0 = Dataset::new(Array2::from(vec![[1,2][3,4]]), Array1::from([0,1]));
/// let ds1 = ds.clone();
///
/// let minmax = normalize(&mut ds0, &None);
/// normalize(&mut ds1, &minmax);
///
/// assert_eq!(ds0, ds1);
/// ```
pub fn normalize(
    dataset: &mut Dataset<f64, (), Ix1>,
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

/// Reads a list of [candump](https://github.com/linux-can/can-utils) log CSV file into a vector of [packets][ids::Packet].  
///
/// The log file must have the following format:  
/// *Timestamp,ID,DLC,Data,Label*  
/// *1478195722.758421,0430,8,00,00,00,00,00,00,00,00,Normal*  
/// *1478195722.766397,02b0,5,ff,7f,00,05,2f,Normal*  
/// *1478195722.769240,0350,8,05,20,74,68,78,00,00,41,Normal*  
/// *1478195722.775132,00df,8,8c,ab,f2,26,7a,29,1a,0c,Attack*  
/// *1478195722.775957,06ea,8,25,10,9c,ed,5b,16,2c,18,Attack*  
///
/// # Examples
/// ```
/// use std::path::Path;
///
/// let packets = packets_from_csv(vec![Path::new("log.csv")])
/// ```
pub fn packets_from_csv(paths: Vec<&Path>) -> Result<Vec<ids::Packet>, csv::Error> {
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

            packets.push(ids::Packet::new(timestamp, id, bytes.to_vec(), flag));
        }
    }
    Ok(packets)
}
