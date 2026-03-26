# Artifacts for revealing the influence of participant failures
This repository provides additional information about the experiments of the paper "Revealing the influence of participant failures on model quality in cross-silo Federated Learning".

It consists of the following artifacts:
- Data Distributions for the different datasets
- Implementation of the data partitioning method
- Implementation details of the model architectures
- Extensive logs of the experiments

## Structure

| Directory | Description|
|----------|----------|
| .data | Contains information about the datasets |
| .experiments | Contains information about the experiments |
| .experiments/images | Resulting visualizations from the experiments |
| .experiments/logs | Logs of all the experiments that were conducted |
| .models | Contains implementation details for different ML models |
| .partitioner | Contains implementation details for the data partitioner |


## Requirements for running the data partitioning
1. Install python (3.12.1)
2. Install the dependencies from the requirements.txt
3. Download the data following the instructions [in the data folder](/data/README.md)
