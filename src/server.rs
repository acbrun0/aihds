pub async fn post(
    client: &reqwest::Client,
    features: Vec<f32>,
    result: &bool,
) -> Result<reqwest::Response, reqwest::Error> {
    client
        .post("http://localhost:9000/")
        .json(&(features, result))
        .send()
        .await
}
