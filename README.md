# AI Content Detection Using BERT and DeBERTa

This research project focuses on the robust detection of AI-generated content (specifically essays) by comparing two state-of-the-art Transformer-based architectures: **BERT** and **DeBERTa**.

##  Project Overview
As Large Language Models (LLMs) continue to evolve, the distinction between human-written and AI-generated text has become a critical challenge for academic integrity. This project explores how deep linguistic features and positional encoding differences between models affect detection accuracy.

##  Model Architectures Compared

### 1. BERT (Bidirectional Encoder Representations from Transformers)
- **Mechanism**: Uses Absolute Position Embeddings.
- **Hidden Size**: 768-dimensional vectors.
- **Max Length**: 512 tokens.
- **Role**: Serves as a strong baseline for bidirectional context understanding.

### 2. DeBERTa (Decoding-enhanced BERT with Disentangled Attention)
- **Mechanism**: Uses **Disentangled Attention** and **Relative Position Bias**.
- **Efficiency**: Separates content and position information into two different vectors, allowing for a more nuanced understanding of syntax.
- **Max Length**: Theoretically capable of handling much longer sequences than BERT.

##  Tech Stack & Methodology
- **Frameworks**: PyTorch, Hugging Face Transformers.
- **Data Pre-processing**: Custom Regex-based cleaning to handle multiple spaces, tabs, and special characters.
- **Training Strategy**: 
    - **Dataset Split**: 70% Training, 15% Validation, 15% Testing.
    - **Optimization**: AdamW optimizer with a linear learning rate scheduler and warmup steps.
    - **Mixed Precision**: Enabled `fp16=True` to optimize GPU VRAM and reduce training time.

##  Key Insights
- **DeBERTa** generally outperforms BERT in detecting mechanical AI patterns due to its superior handling of relative word positions.
- **Learning Rate**: Fine-tuned at `2e-5` to prevent catastrophic forgetting of pre-trained knowledge.

##  How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/professor4044/AI-Content-Detection-Using-BERT-and-DeBERTa.git](https://github.com/professor4044/AI-Content-Detection-Using-BERT-and-DeBERTa.git)
