# GraphSmileWithCommonSense: Multimodal Emotion Recognition with Commonsense and Graph Reasoning

This project combines the strengths of commonsense reasoning and multimodal graph-based learning to enhance emotion 
recognition in conversations. Specifically, we integrate COMET-generated commonsense inferences from the 
COSMIC model with the heterogeneous graph architecture of GraphSmile. The resulting model fuses 
text, audio, and visual modalities, encodes role-specific context using GRU-based commonsense encoders, 
and applies gated fusion and graph convolution to capture emotion dynamics across speakers. Tested on the 
MELD dataset, this hybrid architecture improves emotion classification by leveraging both contextual commonsense and 
intermodal structure.

---

## Base Models

This project is built upon the following two foundational models. 

### COSMIC: COmmonSense knowledge for eMotion Identification in Conversations 
**D. Ghosal, N. Majumder, A. Gelbukh, R. Mihalcea, & S. Poria.**  
[Link] https://arxiv.org/abs/2010.02795 and https://github.com/declare-lab/conv-emotion

### Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition
**J. Li, X. Wang, Z. Zeng**
[Link] https://doi.org/10.1109/TPAMI.2025.3581236 and https://github.com/lijfrank/GraphSmile