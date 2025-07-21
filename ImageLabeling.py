import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

# --- Configuration ---
MODEL_NAME = "google/mt5-small" # Un modèle multilingue, 'mt5-small' est un bon point de départ
OUTPUT_DIR = "./ankamantatra_finetuned_model"
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# --- 1. Chargement et Préparation du Jeu de Données ---
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

print(f"Nombre d'exemples d'entraînement : {len(dataset_dict['train'])}")
print("Exemple de données brutes :")
print(dataset_dict['train'][0])

# --- 2. Chargement du Tokenizer et du Modèle ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- 3. Tokenisation des Données ---
# Fonction pour préparer les données pour le modèle Seq2Seq
def preprocess_function(examples):
    inputs = [f"ankamantatra: {q}" for q in examples["ankamantatra"]]
    targets = [ans for ans in examples["reponse"]]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Appliquer la fonction de préprocessing au dataset
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, remove_columns=["ankamantatra", "reponse"])

print("\nExemple de données tokenisées :")
print(tokenized_datasets['train'][0])

# --- 4. Configuration de l'entraînement ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="no", # Pas d'évaluation sur un jeu de validation pour l'instant
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True, # Important pour les modèles Seq2Seq
    fp16=torch.cuda.is_available(), # Utiliser fp16 si un GPU est disponible pour accélérer l'entraînement
)

# --- 5. Entraînement du Modèle ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

print("\nDémarrage du fine-tuning...")
trainer.train()
print("Fine-tuning terminé et modèle sauvegardé dans", OUTPUT_DIR)

# --- 6. Sauvegarde du modèle et du tokenizer finetunés ---
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)

