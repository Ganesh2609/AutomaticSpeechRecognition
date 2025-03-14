import os
import torch 
from torch import nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from tqdm import tqdm
from torchmetrics.text import WordErrorRate, CharErrorRate


class ModularTrainer:
    """
    A modular trainer class for training and evaluating speech recognition models.
    
    This class handles the entire training loop including:
    - Per-epoch training and testing
    - Loss calculation and optimization
    - Metrics tracking (loss, WER, CER)
    - Checkpoint saving and loading
    - Training visualization through plots
    
    Attributes:
        model: The neural network model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test/validation data
        loss_fn: Loss function (defaults to CTCLoss)
        optimizer: Optimizer for model parameters
        processor: Text processor for decoding model outputs
        logger: Logging utility for tracking training progress
        device: Device to run training on (GPU or CPU)
        history: Dictionary tracking metrics across epochs
        step_history: Dictionary tracking metrics across steps
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 processor = None,
                 log_path: Optional[str] = './logs/training.log',
                 num_epochs: Optional[int] = 16,
                 checkpoint_path: Optional[str] = './checkpoints',
                 graph_path: Optional[str] = './graphs/model_loss.png',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        """
        Initialize the ModularTrainer with model, data loaders, and training parameters.
        
        Args:
            model: PyTorch model to be trained
            train_loader: DataLoader containing training data
            test_loader: DataLoader containing test/validation data
            loss_fn: Loss function for training (defaults to CTCLoss)
            optimizer: Optimizer for updating model parameters (defaults to Adam)
            processor: Text processor for decoding model predictions
            log_path: Path to save training logs
            num_epochs: Number of training epochs
            checkpoint_path: Directory to save model checkpoints
            graph_path: Path to save training visualization graphs
            verbose: Whether to print detailed training information
            device: Device to run training on (auto-detects if None)
        """
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn or nn.CTCLoss(blank=1, zero_infinity=True)
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.loss_update_step = 4

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')

        self.processor = processor
        self.wer_metric = WordErrorRate()
        self.cer_metric = CharErrorRate()

        self.history = {
            'Training Loss': [],
            'Testing Loss': [],
            'Word Error Rate' : [],
            'Character Error Rate' : []
        }

        self.step_history = {
            'Training Loss': [],
            'Testing Loss': [],
            'Word Error Rate' : [],
            'Character Error Rate' : []
        }


    def update_plot(self) -> None:
        """
        Update and save the training visualization plots.
        
        Creates a 2x2 grid of plots showing:
        - Training loss over steps
        - Testing loss over steps
        - Word error rate over steps
        - Character error rate over steps
        
        The plots are saved to the path specified in self.graph_path.
        """

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Steps [every 50]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(self.step_history['Testing Loss'], color='green', label='Testing Loss')
        axs[0, 1].set_title('Testing Loss')
        axs[0, 1].set_xlabel('Steps [every 50]')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()

        axs[1, 0].plot(self.step_history['Word Error Rate'], color='red', label='Word Error Rate')
        axs[1, 0].set_title('Word Error Rate')
        axs[1, 0].set_xlabel('Steps [every 50]')
        axs[1, 0].set_ylabel('Error Rate')
        axs[1, 0].legend()

        axs[1, 1].plot(self.step_history['Character Error Rate'], color='orange', label='Character Error Rate')
        axs[1, 1].set_title('Character Error Rate')
        axs[1, 1].set_xlabel('Steps [every 50]')
        axs[1, 1].set_ylabel('Error Rate')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close(fig)

        return


    def train_epoch(self) -> None:
        """
        Train the model for one epoch.
        
        Processes all batches in the training data loader, computing loss
        and updating model parameters. Progress is displayed using tqdm and
        training metrics are recorded in the history dictionaries.
        
        Updates the visualization plot periodically based on loss_update_step.
        """

        self.model.train()
        total_loss = 0.0

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                audio = batch['Audio'].to(self.device)
                audio_mask = batch['Audio Mask'].to(self.device)
                input_lengths = batch['Input Lengths'].to(self.device)
                transcription = batch['Transcription'].to(self.device)
                target_lengths = batch['Target Lengths'].to(self.device)

                log_probs = self.model(input_values=audio, attention_mask=audio_mask).logits.log_softmax(dim=-1).transpose(0, 1)
                loss = self.loss_fn(log_probs, transcription, torch.clamp(input_lengths, max=log_probs.shape[0]), target_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.current_step += 1

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Train Loss' : total_loss/(i+1),
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i+1))
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        self.history['Training Loss'].append(train_loss)

        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss}")

        return
    


    def test_epoch(self) -> None:
        """
        Evaluate the model on test data for one epoch.
        
        Processes all batches in the test data loader, computing loss and metrics
        without updating model parameters. Tracks word error rate (WER) and 
        character error rate (CER) in addition to loss.
        
        Updates the visualization plot periodically based on loss_update_step.
        """

        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0
        total_cer = 0.0

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, batch in t:
                
                audio = batch['Audio'].to(self.device)
                audio_mask = batch['Audio Mask'].to(self.device)
                input_lengths = batch['Input Lengths'].to(self.device)
                transcription_text = batch["Transcription Text"]
                transcription = batch['Transcription'].to(self.device)
                target_lengths = batch['Target Lengths'].to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_values=audio, attention_mask=audio_mask).logits
                    log_probs = logits.log_softmax(dim=-1).transpose(0,1)
                    loss = self.loss_fn(log_probs, transcription, torch.clamp(input_lengths, max=log_probs.shape[0]), target_lengths)
                    predicted_ids = torch.argmax(logits, dim=-1)
                
                prediction = self.processor.batch_decode(predicted_ids)
                with open('sample2.txt', 'w', encoding='utf-8') as f:
                    f.write(transcription_text[0] + '\n')
                    f.write(transcription_text[1] + '\n')
                    f.write(prediction[0] + '\n')
                    f.write(prediction[1] + '\n')
                wer = self.wer_metric(prediction, transcription_text)
                cer = self.cer_metric(prediction, transcription_text)

                total_loss += loss.item()
                total_wer += wer
                total_cer += cer

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Test Loss' : total_loss/(i+1),
                    'Word Error Rate' : total_wer/(i+1),
                    'Character Error Rate' : total_cer/(i+1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Testing Loss'].append(total_loss / (i+1))
                    self.step_history['Word Error Rate'].append(total_wer / (i+1))
                    self.step_history['Character Error Rate'].append(total_cer / (i+1))
                    self.update_plot()

        test_loss = total_loss / len(self.test_loader)
        test_wer = total_wer / len(self.test_loader)
        test_cer = total_cer / len(self.test_loader)
        self.history['Testing Loss'].append(test_loss)
        self.history['Word Error Rate'].append(test_wer)
        self.history['Character Error Rate'].append(test_cer)

        if test_loss < self.best_metric:
            self.best_metric = test_loss
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing loss for epoch {self.current_epoch}: {test_loss}")
        self.logger.info(f"Testing word error rate for epoch {self.current_epoch}: {test_wer}\n")
        self.logger.info(f"Testing character error rate for epoch {self.current_epoch}: {test_cer}\n")

        return
    

    def train(self, resume_from: Optional[str]=None) -> None:
        """
        Train the model for the specified number of epochs.
        
        The main training loop that calls train_epoch and test_epoch for each epoch.
        Can resume training from a checkpoint if resume_from is specified.
        
        Args:
            resume_from: Path to a checkpoint file to resume training from.
                         If None, training starts from scratch.
        """
        
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")
        

        for epoch in range(self.current_epoch, self.num_epochs + 1):

            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()
    
            self.save_checkpoint()
        
        return
    
    

    def save_checkpoint(self, is_best:Optional[bool]=False):
        """
        Save the current model state and training progress to a checkpoint file.
        
        Args:
            is_best: If True, the checkpoint is saved as 'best_model.pth'.
                     Otherwise, it's saved with the current epoch number.
        """

        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history' : self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_path, 
                f'model_epoch_{self.current_epoch}.pth'
            )

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")


    def load_checkpoint(self, checkpoint:Optional[str]=None, resume_from_best:Optional[bool]=False):
        """
        Load a saved checkpoint to resume training or evaluation.
        
        Restores model and optimizer states, training history, and counters.
        
        Args:
            checkpoint: Path to the checkpoint file. If None and resume_from_best is True,
                        loads the best model checkpoint.
            resume_from_best: If True, loads the best model checkpoint from checkpoint_path.
        """
        
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            self.history[key] = loaded_history.get(key, self.history[key])

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            self.step_history[key] = loaded_step_history.get(key, self.step_history[key])
        
        return