# GraphSmileWithCommonSense
**Multimodal Emotion Recognition with Commonsense and Graph-based Reasoning**

---

## 🚀 Project Motivation

Understanding human emotion in conversations is a core challenge in multimodal AI: text, audio, and visual cues interweave with implicit commonsense knowledge and speaker-role dynamics. In this project, we explore how to fuse **commonsense inferences** (via the COSMIC model) with a **heterogeneous graph structure** (inspired by GraphSmile) to improve emotion recognition in dialogues.

This aligns with my interest in bridging representation learning, human-centric AI, and structured reasoning paradigms — and demonstrates early work toward my goal of building adaptive, reasoning-aware systems in heterogeneous modalities.

---

## 🧠 Technical Highlights

- **Integration of COMET-/COSMIC-generated commonsense triples** into a multimodal pipeline (text + audio + vision)
- Construction of a **heterogeneous conversation graph**: nodes represent modalities, speaker roles and contextual commonsense; edges model inter- and intra-speaker dynamics
- Use of **gated fusion and graph convolutional layers** to combine modalities + commonsense features
- **GRU-based encoder** for role-specific commonsense sequences, enabling dynamic encoding of speaker context
- End-to-end training and evaluation on the **MELD dataset** for emotion classification, demonstrating measurable improvement over the baseline GraphSmile architecture
- **Modular, clean codebase** (Python) with clear separation of data loading, model definition, training scripts and utilities — easy to extend and reproduce

---

## 🏗 Architecture Overview

This project combines two foundational models:

### Base Models

**1. COSMIC: COmmonSense knowledge for eMotion Identification in Conversations**
_D. Ghosal, N. Majumder, A. Gelbukh, R. Mihalcea, & S. Poria_
COSMIC leverages commonsense knowledge from COMET to enhance emotion recognition by modeling mental states, events, and causal relations in conversations.
- Paper: https://arxiv.org/abs/2010.02795
- Github: https://github.com/declare-lab/conv-emotion

**2. GraphSmile: Tracing intricate cues in dialogue**
_J. Li, X. Wang, Z. Zeng_
GraphSmile uses heterogeneous graph structures to model multimodal emotion dynamics across speakers, capturing both intra-speaker and inter-speaker dependencies.
- Paper: https://doi.org/10.1109/TPAMI.2025.3581236
- Github: https://github.com/lijfrank/GraphSmile

### Our Hybrid Architecture

The integration works as follows:

1. **Multimodal Feature Extraction**: Text, audio, and visual features are extracted for each utterance
2. **Commonsense Inference**: COMET/COSMIC generates commonsense knowledge triples (mental states, intentions, causal relations) for each speaker turn
3. **Commonsense Encoding**: A GRU-based encoder processes the commonsense sequences to create role-specific contextual representations
4. **Graph Construction**: Build a heterogeneous graph with nodes representing:
   - Utterance modalities (text, audio, vision)
   - Speaker roles
   - Commonsense knowledge nodes
5. **Graph-based Fusion**: Graph Convolutional Networks (GCN) propagate information across the graph, capturing both multimodal dependencies and commonsense reasoning
6. **Gated Fusion**: Learnable gates balance the contribution of modal features vs. commonsense signals
7. **Emotion Classification**: Final representations are fed to a classifier for emotion prediction

This architecture addresses key challenges:
- **Context propagation**: Graph structure enables long-range dependencies
- **Emotion shift detection**: Commonsense helps identify triggers and causes
- **Multimodal integration**: Heterogeneous graph naturally handles different modality types
- **Speaker dynamics**: Explicit modeling of inter- and intra-speaker relationships

---

## 📁 Project Structure

```text
GraphSmileWithCommonSense/
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
├── LICENSE                            # License of GraphSmile
├── src/
│   ├── dataloader.py                 # Prepares multimodal features + commonsense vectors
│   ├── commonsense_model.py          # Encodes COMET/COSMIC output using GRU
│   ├── GraphSmile_COSMIC_model.py    # Core graph architecture with commonsense nodes
│   ├── hybrid_model.py               # Fuses modal, commonsense & graph embeddings
│   ├── model.py                      # Base model definitions
│   ├── model_utils.py                # Model utility functions
│   ├── module.py                     # Neural network modules
│   ├── trainer.py                    # Training pipeline
│   └── utils.py                      # General utilities
├── scripts/
│   ├── run.py                        # Main training/evaluation script
│   └── analyze_class_distribution.py # Dataset analysis utilities
└── features/                         # Pre-extracted features (if available)
```

### Key Files Explained

- **[dataloader.py](src/dataloader.py)**: Loads and preprocesses multimodal features (text, audio, vision) and integrates commonsense inference vectors
- **[commonsense_model.py](src/commonsense_model.py)**: Implements GRU-based encoding of COMET/COSMIC commonsense outputs into structured representations
- **[GraphSmile_COSMIC_model.py](src/GraphSmile_COSMIC_model.py)**: Defines the core heterogeneous graph architecture that extends GraphSmile with commonsense reasoning nodes
- **[hybrid_model.py](src/hybrid_model.py)**: Implements the fusion mechanism that combines modal embeddings, commonsense embeddings, and graph-based representations
- **[trainer.py](src/trainer.py)** & **[scripts/run.py](scripts/run.py)**: Training and evaluation pipeline with hyperparameter configurations and logging

---

## 📊 Benchmark & Evaluation

### Performance on MELD Dataset

Our hybrid GraphSmile + COSMIC model achieves competitive results on the MELD emotion recognition benchmark:

| Metric | Score |
|--------|-------|
| **Accuracy** | **67.2%** |
| **Weighted F1-Score** | **66.2%** |
| **F1 on "Disgust" Class** | **32.8%** |

### Key Results

- **Surpasses baseline models**: Our approach outperforms SACL-LSTM and COSMIC baselines
- **Matches state-of-the-art**: Achieves performance comparable to top models (M2FNet, GraphSmile)
- **Strong minority class detection**: Notably reached **32.8% F1** on the minority "disgust" class, evidencing robust detection of subtle, low-frequency emotions that are typically challenging for standard models
- **Balanced performance**: The integration of commonsense reasoning helps maintain strong performance across both frequent and rare emotion classes

### Model Comparison

| Model | Accuracy | Weighted F1 | Notes |
|-------|----------|-------------|-------|
| SACL-LSTM | ~64.0% | ~62.0% | Baseline sequential model |
| COSMIC | ~65.2% | ~64.5% | Commonsense-based model |
| GraphSmile | ~67.0% | ~66.0% | Graph-based multimodal model |
| M2FNet | ~67.5% | ~66.5% | Memory fusion network |
| **Ours (GraphSmile + COSMIC)** | **67.2%** | **66.2%** | **Hybrid approach** |

The results demonstrate that our hybrid architecture successfully combines the strengths of graph-based multimodal modeling and commonsense reasoning, particularly improving performance on challenging minority emotion classes.

---

## How to Use

### Prerequisites

- Python 3.7+
- PyTorch 2.0+

```bash
pip install -r requirements.txt
```

### Prepare the MELD Dataset

1. **Download MELD dataset**: Visit [MELD Dataset](https://affective-meld.github.io/) and download the data
2. **Download pre-extracted features**:
   - [RoBERTa features](https://drive.google.com/file/d/1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr/view?usp=sharing) for text encoding
   - [COMET features](https://drive.google.com/file/d/1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr/view?usp=sharing) for commonsense knowledge
   - Multimodal features (audio + visual) from [GraphSmile repository](https://github.com/lijfrank/GraphSmile)

3. **Place the feature files** in the `features/` directory:
   ```
   features/
   ├── meld_features_roberta.pkl
   ├── meld_features_comet.pkl
   └── meld_multi_features.pkl
   ```

### 3. Training

Train the model on MELD dataset for emotion classification:

```bash
python scripts/run.py \
  --gpu 0 \
  --classify emotion \
  --dataset MELD \
  --epochs 50 \
  --textf_mode graphsmile_cosmic \
  --lr 7e-05 \
  --batch_size 16 \
  --hidden_dim 384 \
  --win 3 3 \
  --heter_n_layers 5 5 5 \
  --drop 0.2 \
  --shift_win 3 \
  --lambd 1.0 0.5 0.2 \
  --balance_strategy subsample
```

### 4. Evaluation

The model will automatically evaluate on the test set after training. Results including accuracy, F1-scores, and confusion matrix will be displayed.

### 5. Inference 

```bash
python scripts/run.py \
  --mode inference \
  --checkpoint path/to/checkpoint.pt \
  --input path/to/conversation_features.pkl
```