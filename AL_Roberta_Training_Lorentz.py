import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, \
    TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import EarlyStoppingCallback
import preprocessing_Asialymph
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class CustomDataCollator:
    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
        attention_mask = torch.stack([sample["attention_mask"] for sample in samples], dim=0)
        labels = torch.stack([sample["labels"] for sample in samples], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def encode_data(tokenizer, occupation, max_length=16):
    encoding = tokenizer(occupation, add_special_tokens=True, max_length=max_length, truncation=True,
                         return_token_type_ids=True, padding='max_length', return_tensors='pt')
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)


def generate_input_dict(tokenizer, occupation, label, max_length=16):
    input_ids, attention_mask = encode_data(tokenizer, occupation, max_length)
    label = torch.tensor(label, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


def train_full_transformer_model():
    pp_data = preprocessing_Asialymph.load_and_preprocess_csv('ALtrainingdata_workshop.csv', min_combined_length=3,
                                                              to_lower=True, to_upper=False, remove_punctuation=True,
                                                              remove_chinese=True, stem=False, only_4digit=True,
                                                              only_exist=True)
    pp_data = pp_data[['isco88', 'combined_text']]
    train_data, test_data = preprocessing_Asialymph.train_split(pp_data)
    train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=3)

    le = LabelEncoder()
    le.fit(pp_data['isco88'].unique())

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    num_labels = len(pp_data['isco88'].unique())

    train_data_dicts = [generate_input_dict(tokenizer, occ, le.transform([label])[0]) for
                        occ, label in
                        zip(train_data['combined_text'],
                            train_data['isco88'].values)]
    val_data_dicts = [generate_input_dict(tokenizer, occ, le.transform([label])[0]) for
                      occ, label in
                      zip(val_data['combined_text'],
                          val_data['isco88'].values)]
    test_data_dicts = [generate_input_dict(tokenizer, occ, le.transform([label])[0]) for
                       occ, label in
                       zip(test_data['combined_text'],
                           test_data['isco88'].values)]

    train_input_ids = torch.stack([sample['input_ids'] for sample in train_data_dicts], dim=0)
    train_attention_mask = torch.stack([sample['attention_mask'] for sample in train_data_dicts], dim=0)
    train_labels = torch.stack([sample['labels'] for sample in train_data_dicts], dim=0)

    val_input_ids = torch.stack([sample['input_ids'] for sample in val_data_dicts], dim=0)
    val_attention_mask = torch.stack([sample['attention_mask'] for sample in val_data_dicts], dim=0)
    val_labels = torch.stack([sample['labels'] for sample in val_data_dicts], dim=0)

    test_input_ids = torch.stack([sample['input_ids'] for sample in test_data_dicts], dim=0)
    test_attention_mask = torch.stack([sample['attention_mask'] for sample in test_data_dicts], dim=0)
    test_labels = torch.stack([sample['labels'] for sample in test_data_dicts], dim=0)

    train_dataset = CustomDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = CustomDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = CustomDataset(test_input_ids, test_attention_mask, test_labels)

    # Train the transformer model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=60,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=0,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_steps=0,
        learning_rate=1e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), ],
    )

    trainer.train()

    # save label encoder
    with open('label_encoder.pickle', 'wb') as f:
        pickle.dump(le, f)

    # save model
    trainer.save_model('full_transformer_model_ISCO-88_AL_Roberta')

    # Test the model
    predictions = trainer.predict(test_dataset)

    # Convert logits to label IDs
    pred_label_ids = torch.argmax(predictions.predictions, axis=1).detach().cpu().numpy()

    # Decode the label IDs to label strings
    pred_labels = le.inverse_transform(pred_label_ids)
    true_labels = test_data['isco88'].values

    # Calculate accuracy
    acc = accuracy_score(true_labels, pred_labels)
    print(f"Test Accuracy: {acc:.4f}")

    # Print a detailed classification report
    print(classification_report(true_labels, pred_labels))

    return model


train_full_transformer_model()
