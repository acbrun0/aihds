#![warn(missing_docs)]

//! This module provides networking functionality like opening a CAN socket and making HTTP requests to a web server.

use socketcan::{CANFilter, CANSocket, SFF_MASK};

/// Open a CAN socket with an optional filter.
/// # Examples
/// ```
/// let socket = open_socket("can0", &Some(vec![123,456]))
/// loop {
///     match socket.read_frame() {
///         Ok(frame) => // Process frame
///         Err(why) => // Manage error
///     }
/// }
/// ```
pub fn open_socket(interface: &str, filter: &Option<Vec<u32>>) -> CANSocket {
    match CANSocket::open(interface) {
        Ok(socket) => {
            if let Some(filter) = filter {
                let mut filters = Vec::new();
                for id in filter {
                    match CANFilter::new(*id, SFF_MASK) {
                        Ok(filter) => filters.push(filter),
                        Err(why) => panic!("Could not create CAN filter: {}", why),
                    };
                }
                socket
                    .set_filter(filters.as_slice())
                    .expect("Could not set filter on CAN socket")
            }
            socket
        }
        Err(why) => panic!("Could not open CAN socket: {}", why),
    }
}

/// Make a POST request to a HTTP server.
///
/// This function makes an asyncronous HTTP POST request to a specified URL using the [reqwest] library. It allows the communication of an alarm and the extracted features, as well as an optional message.
/// # Examples
/// ```
/// let client = reqwest::Client::new();
/// post(
///     &client,
///     "localhost:1337",
///     vec![0.0, 1.0, 0.0],
///     &true,
///     Some("Intrusion detected!")
/// )
/// ```
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
