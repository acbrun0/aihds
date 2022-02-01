use std::error::Error;
use std::process;
use serde::Deserialize;
use std::char;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Packet {
    timestamp: f64,
    id: String,
    dlc: u8,
    data0: String,
    data1: String,
    data2: String,
    data3: String,
    data4: String,
    data5: String,
    data6: String,
    data7: String,
    flag: char
}

fn explore_dataset(path: &Path) -> Result<f32, Box<dyn Error>> {
    let mut total = 0.0;
    let mut attacks = 0.0;
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(path)?;

    for result in reader.deserialize() {
        match result {
            Ok(record) => {
                let packet: Packet = record;
                if packet.flag == 'T' { attacks += 1.0 }
                total += 1.0;
            },
            Err(_) => ()
        }
    }
    
    Ok(attacks / total)
}

fn main() {
    let result = explore_dataset(Path::new("datasets/car hacking/RPM_dataset.csv"));
    match result {
        Ok(value) => println!("{:.3}% of packets are attack packets.", value * 100.0),
        Err(err) => {
            println!("{}", err);
            process::exit(1);
        }
    }
}
