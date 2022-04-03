use socketcan::{CANSocket, CANSocketOpenError};

pub fn open_socket(interface: &str) -> Result<CANSocket, CANSocketOpenError> {
    CANSocket::open(interface)
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
