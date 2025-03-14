import os
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoProcessor
import re


class KannadaSpeechDataset(Dataset):
    """
    A custom PyTorch Dataset for loading Kannada speech data.

    Args:
        root_dir (str): Root directory containing 'audio_files' and 'trans_files' subdirectories.
        chars_to_ignore (str): Regex pattern for characters to ignore in transcriptions.
        sampling_rate (int): Desired sampling rate for audio files.

    Attributes:
        audio_dir (str): Path to the audio files directory.
        trans_dir (str): Path to the transcription files directory.
        chars_to_ignore_regex (Pattern): Compiled regex pattern for ignored characters.
        sampling_rate (int): Sampling rate to which audio will be resampled.
        data (list): List of tuples containing (audio file path, cleaned transcription).
    """

    def __init__(self, root_dir, chars_to_ignore='[\,\?\.\!\-\;\:\"\“\%\‘\”\।\’\']', sampling_rate=16000):
        super(KannadaSpeechDataset, self).__init__()
        self.audio_dir = os.path.join(root_dir, "audio_files")
        self.trans_dir = os.path.join(root_dir, "trans_files")
        self.chars_to_ignore_regex = re.compile(chars_to_ignore)
        self.sampling_rate = sampling_rate
        self.data = []

        audio_files = os.listdir(self.audio_dir)

        for file in audio_files:
            if file.endswith(".wav"):
                trans_file = os.path.join(self.trans_dir, file.replace(".wav", ".txt"))
                if os.path.exists(trans_file):
                    with open(trans_file, "r", encoding="utf-8") as f:
                        transcription = f.read().strip()
                    # Clean the transcription by removing unwanted characters and converting to lowercase
                    transcription = re.sub(self.chars_to_ignore_regex, '', transcription).lower()
                    self.data.append((os.path.join(self.audio_dir, file), transcription))

    def __len__(self):
        """
        Returns the number of audio-transcription pairs in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the audio waveform and corresponding transcription for a given index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: (waveform, transcription) where:
                - waveform (list): List of audio samples.
                - transcription (str): Cleaned transcription.
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        audio_path, transcription = self.data[idx]
        
        # Load audio and resample if necessary
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_rate)

        # Remove channel dimension if it exists
        waveform = waveform.squeeze(0)
        
        return [list(waveform), transcription]


def collate_function(batch):
    """
    Custom collate function for combining audio and transcription data into a batch.

    Args:
        batch (list): List of samples where each sample is a tuple (waveform, transcription).

    Returns:
        dict: Dictionary containing:
            - 'Audio' (Tensor): Batched audio input.
            - 'Audio Mask' (Tensor): Attention mask for the audio input.
            - 'Input Lengths' (Tensor): Length of each input audio sequence.
            - 'Transcription Text' (list): List of transcriptions as strings.
            - 'Transcription' (Tensor): Flattened and encoded transcriptions.
            - 'Target Lengths' (Tensor): Length of each encoded transcription.
    """
    processor = AutoProcessor.from_pretrained("TheAIchemist13/kannada_beekeeping_wav2vec2")

    waveform = [i[0] for i in batch]
    input_lengths = torch.tensor([len(w) // 320 for w in waveform], dtype=torch.long)
    transcription = [i[1] for i in batch]
    sampling_rate = 16000

    # Process audio input using the processor
    inputs = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    audio = inputs.input_values
    audio_mask = inputs.attention_mask

    # Encode transcription
    labels = processor(text=transcription)["input_ids"]
    flattened_labels = torch.cat([torch.tensor(l, dtype=torch.long) for l in labels])
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return {
        "Audio": audio,
        "Audio Mask": audio_mask,
        "Input Lengths": input_lengths,
        "Transcription Text": transcription,
        "Transcription": flattened_labels,
        "Target Lengths": target_lengths,
    }


def get_data_loaders(root: str, batch_size: int = 16, num_workers: int = 12, prefetch_factor: int = 2):
    """
    Creates PyTorch DataLoaders for Kannada speech data.

    Args:
        root (str): Root directory containing the dataset.
        batch_size (int, optional): Number of samples per batch. Default is 16.
        num_workers (int, optional): Number of worker threads for loading data. Default is 12.
        prefetch_factor (int, optional): Number of samples to prefetch per worker. Default is 2.

    Returns:
        tuple: (train_loader, test_loader) where:
            - train_loader (DataLoader): DataLoader for training data.
            - test_loader (DataLoader): DataLoader for test data.
    """
    data = KannadaSpeechDataset(root)
    torch.manual_seed(42)
    train_size = int(0.8 * len(data))
    train_data, test_data = torch.utils.data.random_split(data, [train_size, len(data) - train_size])

    # Create train and test DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_function
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_function
    )

    return train_loader, test_loader
