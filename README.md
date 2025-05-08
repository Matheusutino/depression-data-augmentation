# Synthetic Data for Mental Health: A Comparative Analysis of LLMs, BERT, and Copy-Based Augmentation

This repository includes the source code, augmentation templates, and experimental configurations for the paper **"Synthetic Data for Mental Health: A Comparative Analysis of LLMs, BERT, and Copy-Based Augmentation"**, submitted to BRACIS 2025.

## 🧠 Overview

We investigate textual data augmentation strategies to improve depression screening from Brazilian-Portuguese Instagram posts. The following methods were evaluated:

- ✅ Simple duplication (Copy-based augmentation)
- ✅ Contextual word substitution using BERT-based models
- ✅ Synthetic text generation via Large Language Models (LLMs), with and without modulation using BDI-II (Beck’s Depression Inventory) psychometric data

Classification is performed using:

- 📘 Single-Instance Learning (SIL)
- 📗 Multiple-Instance Learning (MIL)

All models are trained on multilingual sentence embeddings and evaluated using an XGBoost classifier with Bayesian hyperparameter optimization.

## 🔍 Key Findings

- 🥇 MIL consistently outperforms SIL in all augmentation scenarios
- 🧠 LLM-based augmentation without BDI-II modulation achieves the best F1-scores
- ⚡ Contextual BERT substitution offers a strong performance–efficiency tradeoff
- ⚠️ Synthetic generation with BDI-II introduces stylistic divergence and reduces performance

## 📊 Dataset

The experiments use a real Instagram dataset with BDI-II clinical annotations, collected from university students in Brazil.

- 221 users
- Posts labeled as depressed or non-depressed
- BDI-II cutoffs: minimal/mild (non-depressed), moderate/severe (depressed)
- Only training data is augmented; validation and test sets remain unaltered

## 📁 Repository Structure

- configs/ # Configuration files for prompts 
- src/core/classifier/ # Implementations for SIL and MIL classifiers
- src/core/llm_predictor/ # Wrapper classes for different LLM providers
- src/core/preprocessing.py # Text cleaning and preparation functions
- src/core/utils.py # General-purpose utility functions
- src/scripts/classifier.py # Run model training and evaluation
- src/scripts/data_augmentation.py # Apply text data augmentation strategies
- supplementary_texts/ # Supporting documents for the paper


## 📎 Supplementary Materials

- [Prompt examples used for data generation](./supplementary_texts/prompts.pdf)
- XGBoost hyperparameter search space
- [Evaluation plots and similarity metrics (cosine similarity, Hausdorff distance, etc.)](./supplementary_texts/similarity.pdf)

## 🔬 Citation

Coming soon (after peer review).

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or collaboration inquiries, please contact:  
[REMOVED FOR DOUBLE BLIND REVIEW]

