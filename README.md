---
title: Political Ideologies Analysis and Classification
emoji: 🍃
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: A Political Ideologis and Issues Analysis and Classification using Bidirectional LSTM
---

# Political Ideologies Analysis

This project provides a comprehensive analysis of political ideologies using data from the Huggingface Political Ideologies dataset. The analysis involves data preprocessing, mapping ideological labels, and visualizing political statements through Word2Vec embeddings and t-SNE projections. Additionally, an interactive tool is created for exploring political ideologies and their related issue types in a 2D space.

## Project Overview

The goal of this project is to analyze the political ideologies dataset to understand the distribution of political ideologies (conservative vs liberal) and their association with various issue types. The analysis involves:

- **Data Loading and Cleaning**: Loading, cleaning, and mapping data from the Huggingface dataset.
- **Label Mapping**: Mapping ideological labels (conservative and liberal) and issue types to numerical values.
- **Word2Vec Embeddings**: Generating word embeddings for political statements to create vector representations.
- **Dimensionality Reduction**: Using t-SNE to reduce the dimensionality of embeddings and visualize them in 2D.
- **Interactive Visualizations**: Visualizing the data using Altair with interactive charts to explore ideology and issue type distributions.

## Dataset

The dataset used in this project is the [Political Ideologies dataset](https://huggingface.co/datasets/JyotiNayak/political_ideologies) from Huggingface, which contains political statements along with their corresponding labels (conservative or liberal) and issue types (economic, environmental, social, etc.).

## Requirements

- Python 3.x
- TensorFlow
- Gensim
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Altair

You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```
