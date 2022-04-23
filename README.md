# Automated and Intelligent Hacking Detection System - A machine-learning based IDS for the Controller Area Network.

### Requirements

- Rust
- NodeJS
- npm

### Instructions

#### IDS

##### Build

`cargo build --release --target=armv7-unknown-linux-gnueabihf`

##### Run

```
USAGE:
    aihds [OPTIONS]

OPTIONS:
        --extract-features         Extracts features to CSV files
        --grid-search              Perform grid search optimization on SVM
    -h, --help                     Print help information
        --join                     Join features into a single file
        --libsvm                   Use libsvm format
        --live                     Run IDS in live mode
        --model <MODEL>            Path to model to be loaded
        --monitor <MONITOR>        IDs to monitor
        --streaming <STREAMING>    Run model in streaming mode
        --test <TEST>              Paths to the datasets required for testing the model, separated
                                   by ','
        --train <TRAIN>            Paths to the datasets required for training the model, separated
                                   by ','
    -V, --version                  Print version information
```

##### Example

Run in live mode with a saved model and show data in a server located at _url:port_  
`./aihds --live --model models/svm --streaming [url:port]`  
Simulate blinker attack  
`cangen can0 -I 0ca -D D7260101FFFFB9E7 -g 10 -L 8`

#### Website

Inside the _aihds/webapp_ directory, run `npm i` to install dependencies and then `npm start` to execute the server.

## Datasets

Datasets used can be found [here](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset), [here](https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset) and [here](https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2).