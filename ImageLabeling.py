import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

# --- Configuration ---
MODEL_NAME = "google/mt5-small" # Un mod�le multilingue, 'mt5-small' est un bon point de d�part
OUTPUT_DIR = "./ankamantatra_finetuned_model"
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# --- 1. Chargement et Pr�paration du Jeu de Donn�es ---
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

raw_data = load_data('ankamantatra_dataset.json')

# Convertir en format Hugging Face Dataset
# Pour un petit dataset, on peut le mettre directement dans le train split
dataset_dict = DatasetDict({
    'train': Dataset.from_list(raw_data)
})

print(f"Nombre d'exemples d'entra�nement : {len(dataset_dict['train'])}")
print("Exemple de donn�es brutes :")
print(dataset_dict['train'][0])

# --- 2. Chargement du Tokenizer et du Mod�le ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- 3. Tokenisation des Donn�es ---
# Fonction pour pr�parer les donn�es pour le mod�le Seq2Seq
def preprocess_function(examples):
    inputs = [f"ankamantatra: {q}" for q in examples["ankamantatra"]]
    targets = [ans for ans in examples["reponse"]]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Appliquer la fonction de pr�processing au dataset
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, remove_columns=["ankamantatra", "reponse"])

print("\nExemple de donn�es tokenis�es :")
print(tokenized_datasets['train'][0])

# --- 4. Configuration de l'entra�nement ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="no", # Pas d'�valuation sur un jeu de validation pour l'instant
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True, # Important pour les mod�les Seq2Seq
    fp16=torch.cuda.is_available(), # Utiliser fp16 si un GPU est disponible pour acc�l�rer l'entra�nement
)

# --- 5. Entra�nement du Mod�le ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

print("\nD�marrage du fine-tuning...")
trainer.train()
print("Fine-tuning termin� et mod�le sauvegard� dans", OUTPUT_DIR)

# --- 6. Sauvegarde du mod�le et du tokenizer finetun�s ---
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)

