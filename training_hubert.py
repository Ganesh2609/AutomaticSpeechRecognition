import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from trainer import ModularTrainer
from dataset_hubert import get_data_loaders
from transformers import AutoProcessor, HubertForCTC

def main():
    """
    Fine-tunes the HuBERT model on Kannada speech data using CTC loss.

    Steps:
    1. Loads the pre-trained HuBERT model and processor.
    2. Loads the dataset and creates data loaders for training and testing.
    3. Defines the optimizer and loss function (CTCLoss).
    4. Initializes the ModularTrainer with training parameters.
    5. Starts training or resumes from a checkpoint.

    The training process logs results, saves model checkpoints, and generates performance graphs.
    """

    # Select device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained HuBERT processor and model
    processor = AutoProcessor.from_pretrained("TheAIchemist13/kannada_beekeeping_wav2vec2")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft", pad_token_id=processor.tokenizer.pad_token_id)

    # Define dataset root path
    root_path = "E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Datasets/mile_kannada"
    
    # Load training and testing data
    train_loader, test_loader = get_data_loaders(root=root_path, batch_size=2)

    # Define optimizer and loss function
    learning_rate = 1e-5
    loss_fn = nn.CTCLoss(blank=1, zero_infinity=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Initialize the trainer
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        processor=processor,
        log_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Logs/hubert_train_1.log',
        num_epochs=16,
        checkpoint_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Checkpoints/hubert train 1',
        graph_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Graphs/hubert_train_1.png',
        verbose=True,
        device=device
    )

    # Start training
    trainer.train()
    # trainer.train(resume_from="E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Checkpoints/hubert train 1/model_epoch_7.pth")

if __name__ == '__main__':
    main()
