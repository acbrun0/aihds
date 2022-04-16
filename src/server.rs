use socketcan::{CANFilter, CANSocket};

pub fn open_socket(interface: &str, filter: &Option<Vec<u32>>) -> CANSocket {
    match CANSocket::open(interface) {
        Ok(socket) => {
            if let Some(filter) = filter {
                let mut filters = Vec::new();
                for id in filter {
                    match CANFilter::new(*id, 0) {
                        Ok(filter) => filters.push(filter),
                        Err(why) => panic!("Could not create CAN filter: {}", why),
                    };
                }
                socket.set_filter(filters.as_slice()).expect("Could not set filter on CAN socket")
            }
            socket
        }
        Err(why) => panic!("Could not open CAN socket: {}", why),
    }
}

pub async fn post(
    client: &reqwest::Client,
    url: &str,
    features: Vec<f32>,
    result: &bool,
    message: Option<String>,
) -> Result<reqwest::Response, reqwest::Error> {
    client
        .post(format!("http://{}", url))
        .json(&(features, result, message))
        .send()
        .await
}
