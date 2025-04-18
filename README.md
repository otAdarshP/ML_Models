# ML_Models

A Jupyter Notebook-based project comparing four popular machine learning models—Support Vector Machine (SVM), Decision Tree, Random Forest, and Logistic Regression—for sentiment classification on a synthetic dataset of over 1,000 samples.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates a comparative analysis of four machine learning algorithms for sentiment prediction. By running the steps in the provided Jupyter Notebook, you can:

- Load and explore a synthetic sentiment dataset.
- Preprocess text data using TF-IDF vectorization.
- Train and evaluate SVM, Decision Tree, Random Forest, and Logistic Regression models.
- Compare their performance using accuracy, precision, recall, F1‑score, and visualizations.

## Dataset

The dataset is synthetically generated within the notebook (or loaded from a bundled file) and contains:

- **Samples**: ~1,000 text entries representing sentiment-labeled phrases.
- **Features**: Text data vectorized with TF-IDF.
- **Labels**: Binary sentiment classes (e.g., positive/negative).

> _If you have your own dataset, modify the data-loading cell to point to your file._

## Methodology

### Data Preprocessing

1. **Loading Data**: Read the dataset into a Pandas DataFrame.  
2. **Cleaning**: Remove punctuation, lowercase conversion, and optional stop‑word removal.  
3. **Vectorization**: Transform text into numeric features using `sklearn.feature_extraction.text.TfidfVectorizer`.

### Model Training

For each algorithm:

- Split data into training and test sets (e.g., 80/20 split).
- Instantiate the model with default (or specified) hyperparameters.
- Fit the model on the training data.

### Evaluation Metrics

- **Accuracy**: Overall fraction of correctly classified samples.  
- **Precision & Recall**: For positive class, measures correctness and completeness.  
- **F1‑Score**: Harmonic mean of precision and recall.  
- **Confusion Matrix**: Visual overview of true vs. predicted labels.

## Project Structure

```
ML_Models/
├── ML_Model_Comparison.ipynb   # Main notebook with code and analysis
└── README.md                   # Project documentation
```

## Dependencies

- Python 3.7+  
- Jupyter Notebook  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/otAdarshP/ML_Models.git
   cd ML_Models
   ```
2. **(Optional) Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

## Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `ML_Model_Comparison.ipynb` in your browser.
3. Run cells sequentially to reproduce data loading, model training, and evaluation.

## Results

After execution, you’ll see:

- Tabular performance metrics for each model.  
- Bar charts comparing accuracy and F1‑scores.  
- Confusion matrices highlighting classification errors.

Use these insights to understand which algorithm performs best on your sentiment data.

## Contributing

Contributions welcome! To contribute:

1. Fork this repository.  
2. Create a branch: `git checkout -b feature/YourFeature`.  
3. Commit changes: `git commit -m "Add <feature>"`.  
4. Push to your fork: `git push origin feature/YourFeature`.  
5. Open a Pull Request.

## License

_No license specified. You may add one by including a LICENSE file in the project root._

