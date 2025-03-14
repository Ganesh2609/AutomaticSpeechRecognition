import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from trainer import ModularTrainer
from dataset_wav2vec import get_data_loaders
from transformers import AutoProcessor, AutoModelForCTC

def main():
    """
    Fine-tunes the Wav2Vec2-XLS-R-53 model on Tamil speech data using CTC loss.

    Steps:
    1. Loads the pre-trained Wav2Vec2-XLS-R-53 model and processor.
    2. Loads the dataset and creates data loaders for training and testing.
    3. Defines the optimizer and loss function (CTCLoss).
    4. Initializes the ModularTrainer with training parameters.
    5. Resumes training from a saved checkpoint or starts fresh.

    The training process logs results, saves model checkpoints, and generates performance graphs.
    """

    # Select device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained Wav2Vec2 processor and model
    processor = AutoProcessor.from_pretrained("anuragshas/wav2vec2-xlsr-53-tamil")
    model = AutoModelForCTC.from_pretrained("anuragshas/wav2vec2-xlsr-53-tamil")

    # Define dataset root path
    root_path = "E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Datasets/mile_tamil"
    
    # Load training and testing data
    train_loader, test_loader = get_data_loaders(root=root_path, batch_size=4)

    # Define optimizer and loss function
    learning_rate = 1e-5
    loss_fn = nn.CTCLoss(blank=49, zero_infinity=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Initialize the trainer
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        processor=processor,
        log_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Logs/wav2vec2_train_2.log',
        num_epochs=32,
        checkpoint_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Checkpoints/wav2vec2 train 2',
        graph_path='E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Graphs/wav2vec2_train_2.png',
        verbose=True,
        device=device
    )

    # Resume training from a specific checkpoint
    trainer.train(resume_from="E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Checkpoints/wav2vec2 train 2/model_epoch_7.pth")

if __name__ == '__main__':
    main()
