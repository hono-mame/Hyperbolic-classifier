# How to Run the Code
## Dataloading.hs
This script extracts the hypernym-hyponym tree structure from the WordNet database (/data/wnjpn.db) and saves it in various formats.
```bash
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Research && stack run DataLoading" 
```

## Preprocess.hs
Based on the constructed tree structure, this script takes two word inputs and outputs the path and distance between them.
```bash
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Research && stack run Preprocess"
```
```haskell
Enter Node A:
猛打
Enter Node B:
気づまりさ
Path A: entity -> 擲る -> 猛打
Path B: entity -> 面映ゆさ -> 気づまりさ
Distance between 猛打 and 気づまりさ: 4
```

## GenerateDatasetMLP.hs
This script creates a dataset for MLP training and evaluation from the .edges file generated by Dataloading.hs.
```bash
docker-compose exec hasktorch /bin/bash -c "cd /home/ubuntu/Research && stack run GenerateDatasetMLP"
```