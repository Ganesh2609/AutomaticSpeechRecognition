# Speech Recognition with HuBERT, Wav2Vec2, and Whisper

This project focuses on fine-tuning state-of-the-art speech recognition models (HuBERT, Wav2Vec2, and Whisper) for Tamil and Kannada languages using CTC loss. The implementation includes a modular training framework with comprehensive logging, metrics tracking, and visualization capabilities.

## Project Overview

The project implements:
- **HuBERT model fine-tuning** for Kannada speech recognition
- **Wav2Vec2 model fine-tuning** for Tamil speech recognition  
- **Whisper model fine-tuning** on Common Voice Tamil dataset
- Modular training framework with checkpoint management
- Comprehensive evaluation metrics (WER and CER)
- Real-time training visualization

## Architecture

### Models Used
1. **HuBERT (Hidden-Unit BERT)**: Facebook's HuBERT Large model fine-tuned for Kannada
2. **Wav2Vec2-XLS-R-53**: Multilingual Wav2Vec2 model fine-tuned for Tamil
3. **Whisper**: OpenAI's Whisper base model fine-tuned for Tamil transcription

### Key Components
- `trainer.py`: Modular training framework with checkpoint management
- `dataset_hubert.py` & `dataset_wav2vec.py`: Custom dataset loaders for Kannada and Tamil
- `finetune_whisper.py`: Whisper model fine-tuning pipeline
- `logger.py`: Comprehensive logging utility with rotation

## Results

### Training Performance

#### Wav2Vec2 Training (Tamil)
![Wav2Vec2 Training](Train%20Data/Graphs/wav2vec2_train_2.png)

#### HuBERT Training (Kannada)
![HuBERT Training](Train%20Data/Graphs/hubert_train_1.png)

#### Whisper Training - Loss Curve
![Loss Curve](Train%20Data/Graphs/q2_loss.png)

#### Whisper - Word Error Rate (WER) & Character Error Rate (CER)
![Word Error Rate (WER) & Character Error Rate (CER)](Train%20Data/Graphs/q2_wer_cer.png)

## Technical Details

### Dataset Processing
- Audio files resampled to 16kHz
- Transcriptions cleaned using regex patterns
- 90/10 train-test split for Wav2Vec2
- 80/20 train-test split for HuBERT
- 1000 training samples for Whisper fine-tuning

### Training Configuration
- **Optimizer**: Adam with learning rates between 1e-5 and 1e-4
- **Loss Function**: CTC Loss with appropriate blank token indices
- **Batch Sizes**: 2-16 depending on model and GPU memory
- **Epochs**: 16-32 epochs with early stopping based on validation loss

### Evaluation Metrics
- **Word Error Rate (WER)**: Measures word-level transcription accuracy
- **Character Error Rate (CER)**: Measures character-level transcription accuracy
- **Loss Curves**: Track training and validation losses over epochs

## Code Structure

```
project/
├── trainer.py              # Modular training framework
├── dataset_hubert.py       # Kannada dataset loader
├── dataset_wav2vec.py      # Tamil dataset loader
├── training_hubert.py      # HuBERT training script
├── training_wav2vec.py     # Wav2Vec2 training script
├── finetune_whisper.py     # Whisper fine-tuning script
├── logger.py               # Logging utilities
└── Train Data/
    ├── Graphs/             # Training visualizations
    ├── Logs/               # Training logs
    └── Checkpoints/        # Model checkpoints
```

## Key Features

1. **Modular Training Framework**
   - Checkpoint saving/resuming
   - Real-time metric tracking
   - Automatic best model selection
   - Comprehensive logging

2. **Multi-Model Support**
   - Support for HuBERT, Wav2Vec2, and Whisper architectures
   - Model-specific data preprocessing
   - Flexible loss function configuration

3. **Visualization and Monitoring**
   - Real-time plotting of training metrics
   - Step-wise and epoch-wise tracking
   - Automatic graph generation

## Requirements

- PyTorch
- Transformers
- torchaudio
- torchmetrics
- datasets
- evaluate
- matplotlib
- scikit-learn
- tqdm

## Performance Summary

- **HuBERT (Kannada)**: Achieved best test loss of 0.508 with WER ~0.70
- **Wav2Vec2 (Tamil)**: Achieved best test loss of 0.413 with WER ~0.65
- **Whisper (Tamil)**: Demonstrated consistent improvement in WER/CER over training
