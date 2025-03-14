# Importing necessary libraries
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import matplotlib.pyplot as plt

# Load and preprocess the Common Voice dataset for Tamil, selecting a subset and removing unnecessary columns
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "ta", split="train+validation").select(range(1000))
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "ta", split="test").select(range(200))
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Load Whisper components (feature extractor, tokenizer, processor, model) for Tamil transcription using the base model
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="ta", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ta", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Set generation configuration for Tamil transcription and disable forced decoder IDs
model.generation_config.language = "ta"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Prepare dataset by extracting input features from audio and tokenizing labels for training
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch 
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

# Define a custom data collator for padding input features and labels for speech-to-text training
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
    
# Initialize the data collator with the processor and decoder start token ID
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define WER and CER metrics computation for model evaluation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# Set training arguments for fine-tuning Whisper, including batch size, learning rate, and evaluation strategy
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-base-ta",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Initialize the trainer with model, datasets, collator, metrics, and training arguments
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# training the model
trainer.train()

# Plot training and validation loss over steps for performance visualization
logs = trainer.state.log_history
train_loss = [entry["loss"] for entry in logs if "loss" in entry]
train_steps = [entry["step"] for entry in logs if "loss" in entry]
eval_loss = [entry["eval_loss"] for entry in logs if "eval_loss" in entry]
eval_steps = [entry["step"] for entry in logs if "eval_loss" in entry]
plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_loss, label="Train Loss", linestyle="-")
plt.plot(eval_steps, eval_loss, label="Validation Loss", marker="s", linestyle="--")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Plot WER and CER over evaluation steps to track model accuracy
wer = [entry["eval_wer"] for entry in logs if "eval_wer" in entry]
cer = [entry["eval_cer"] for entry in logs if "eval_cer" in entry]
eval_steps = list(range(1, len(wer) + 1))
plt.figure(figsize=(8, 5))
plt.plot(eval_steps, wer, label="WER", marker="o", linestyle="--")
plt.plot(eval_steps, cer, label="CER", marker="s", linestyle="--")
plt.xlabel("Evaluation Steps")
plt.ylabel("Error Rate")
plt.title("WER and CER over Time")
plt.legend()
plt.grid()
plt.show()