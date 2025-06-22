# Meta Recommendation System

## Overview

The **Meta Recommendation System** is a novel framework designed to automate the selection of the most suitable recommendation algorithm for any given dataset. Instead of recommending products to users, this system recommends the optimal recommendation approach (e.g., collaborative filtering, content-based, hybrid) by analyzing the dataset’s characteristics. The aim is to minimize manual experimentation, improve efficiency, and enhance the performance of recommendation systems in real-world applications by leveraging deep learning, semantic code embeddings, and engineered dataset features[1][2].

---

## Table of Contents

- [Motivation](#motivation)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Datasets & Dependencies](#datasets--dependencies)
- [Usage](#usage)
- [Results](#results)
- [Why the Model Works](#why-the-model-works)
- [References](#references)

---

## Motivation

Selecting the right recommendation algorithm for a dataset is a complex, time-consuming process that typically involves trial-and-error. The Meta Recommendation System addresses this by learning to map dataset features to the most semantically and operationally appropriate recommendation algorithm, automating and optimizing this crucial step in the machine learning pipeline[1].

---

## How It Works

### 1. **Input-Output Modalities**
- Supports four modalities: Image-to-Image (I2I), Text-to-Image (T2I), Image-to-Text (I2T), and Text-to-Text (T2T).
- Current implementation demonstrates Text-to-Text (T2T) due to dataset and code availability, but the architecture is extensible to all modalities[1].

### 2. **Embedding Generation**
- **Algorithm Embedding:**  
  - The source code or textual description of each candidate recommendation system is embedded using a pre-trained CodeBERT model, producing a 768-dimensional vector that captures deep semantic and architectural features[1][2].
- **Dataset Embedding:**  
  - Each dataset is characterized by a 50-dimensional vector, engineered from structural, statistical, information-theoretic, and complexity features (e.g., number of samples, sparsity, entropy, outlier ratios)[1][2].

### 3. **Neural Network Alignment**
- A deep residual neural network maps the 50D dataset embedding to the 768D algorithm embedding space, using residual connections, normalization, ReLU activations, and dropout for stability and generalization[1][2].
- The network is trained using a contrastive loss function to maximize similarity between correct dataset-algorithm pairs and minimize it for incorrect pairs, enforcing a clear margin for separation[1].

### 4. **Evaluation and Recommendation**
- After training, the model predicts the best-matching recommendation system for new datasets by comparing predicted embeddings to those of candidate algorithms, reporting similarity scores and gaps[1][2].

---

## Project Structure

```
├── Recommendation_recommender_code.ipynb   # Main meta-recommender notebook (core logic)
├── Amazon_Product_Recommendation_system.ipynb
├── Netflix_Recommendation_System.ipynb
├── Spotify_Recommendation_System.ipynb
├── DS_Project_Report_Group2.pdf            # Detailed project report
├── datasets/
│   ├── ratings_Electronics.csv
│   ├── netflix_dataset.csv
│   ├── data.csv (Spotify)
│   └── ...
```
- **Recommendation_recommender_code.ipynb** is the main entry point for the meta-recommendation pipeline.
- Other notebooks are representative recommendation systems used for model training and evaluation[2].

---

## Datasets & Dependencies

- **Datasets Used:**
  - Spotify Music Dataset
  - Netflix Movie Dataset
  - Amazon Product Dataset
  - (See `datasets/` folder for details)[1].
- **Libraries:**
  - PyTorch (deep learning)
  - Hugging Face Transformers (CodeBERT)
  - pandas, numpy (data processing)
  - scikit-learn (feature engineering, evaluation)
  - matplotlib, seaborn (visualization)[1][2].

---

## Usage

### 1. **Installation**

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

### 2. **Running the Meta Recommendation System**

- Open `Recommendation_recommender_code.ipynb` in Jupyter or Google Colab.
- Configure paths to your datasets and code files.
- Run all cells to:
  - Generate embeddings for datasets and algorithms.
  - Train the neural network.
  - Evaluate and visualize recommendations for sample or new datasets.
- To test on new data, use the provided function to input a new dataset and receive the recommended system with similarity scores and confidence gap[2].

---

## Results

- **Performance:**  
  The model achieves high similarity between predicted and correct algorithm embeddings, with significant similarity gaps over incorrect matches, indicating robust and confident recommendations[1][2].
- **Visualization:**  
  Training loss, similarity evolution, and feature importance plots are included for interpretability[1].
- **Case Studies:**  
  The system recommends, for example, the Spotify recommender for Apple Music data with high confidence, and Netflix content-based for movie datasets[1][2].

---

## Why the Model Works

By embedding both dataset features and algorithm representations into a joint vector space, the model enables meaningful interactions and similarity-based reasoning across previously unseen datasets and algorithms. CodeBERT encodes deep architectural patterns from code, while engineered dataset features provide a rich, discriminative characterization of each dataset. The residual neural network captures non-linear, high-dimensional relationships, and the contrastive loss ensures robust, interpretable, and task-aware recommendations. This approach enables automated, scalable, and principled model selection for recommendation tasks[1].

---

## References

For a full list of references and further reading, see the project report (`DS_Project_Report_Group2.pdf`)[1].

---

**Contributors:**  
A. Chaitanya, B. Keerthan, G. Avinash, K. Dinesh Siddhartha, P. Praneeth  
IIT Gandhinagar, Gujarat, India

---

*For detailed methodology, experiments, and results, please refer to the full project report (`DS_Project_Report_Group2.pdf`). All code and datasets are included in this repository.*

