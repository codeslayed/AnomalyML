# AnomalyML: Machine Learning to Detect Credit Card Anomaly

## Overview

AnomalyML is a machine learning project aimed at detecting anomalies in credit card transactions using a dataset from Kaggle. The primary goal of this project is to identify fraudulent transactions in order to enhance security measures in financial systems.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualization](#visualization)
- [Pushing Large Files with Git LFS](#pushing-large-files-with-git-lfs)
- [Contributing](#contributing)
- [License](#license)

## Project Description

This project utilizes machine learning techniques, specifically autoencoders, to detect anomalies in credit card transactions. By training the model on historical transaction data, we aim to identify patterns that indicate fraudulent activity.

## Dataset

The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset includes:

- **Features**: Various anonymized features derived from PCA transformation.
- **Target Variable**: A binary class indicating whether a transaction is fraudulent (1) or not (0).


## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```
## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AnomalyML.git
   cd AnomalyML
   ```
2. Place the creditcard.csv dataset in the project directory.

3. Run the main script:
  ```bash
   python prototype.py
   ```
   
