Hi! My name is Alec Ippolito and I am very interested in the Machine Learning Engineer position at Govini.

I have had the first phone interview and was given this prompt to solve:
Use a subset of the publicly available Google patents dataset to create a binary classifier that
classifies each row as 0 or 1.

It was suggested to use the following libraries:
- Huggingface Transformer
- Huggingface Accelerate
- Huggingface Evaluate
- Polars
- Pandas
- loguru
- Hydra

In my implementation, I used pandas, loguru and hydra libraries. Besides this I used sklearn, numpy, torch

My assignment is compileable through the IDE or through the command line. To run through the command
line, please download all of the source code and run by the following command: 
**python src/main.py**

Since I implemented this with the hydra library, there a way to config this at the command line by:
**python src/main.py parameter_name=value**
The paramter_name options are as follows: paths.path_csv, network.epochs, netowrk.batch_size, network.learning_rate
Here is an example of running through the commmand line with the configs being connfigured by assigning as the default values
**python src/main.py paths.path_csv="new_csv_path.csv" network.epochs=10 network.batch_size=64 network.learning_rate=0.001**

The folder structure of this project should be:
GOVINI TAKEHOME ASSESSMENT
  hydra
    config.yaml
    hydra.yaml
    overrides.yaml
  conf
    network
      network_config.yaml
    paths
      default_paths.yaml
    config.yaml
  src
    {time:YYYY-MM-DD}.log
    data.py
    encode.py
    evaluate.py
    main.py
    network.py
  ML Takehome.pdf
  ml_dataset.csv
