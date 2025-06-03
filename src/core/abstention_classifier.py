from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from src.common.logger import get_logger
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import random
import os


MODEL_NAME = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
HEAD_SAVE_LOCALE = '.local/AbstentionClassifier/BinaryHead/best_classifier.pt'
NUM_WORKERS = 8

logger = get_logger(__name__)


class AbstentionClassifier(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            torch_dtype='auto'
        )

        self.device = self.model.device

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, 1).to(self.model.device, dtype=self.model.dtype)
        self.prompt_template = (
            "You are a data scientist, who has to vet questions from users.\n"
            "You have received the question: \"{question}\" "
            "for the database described by the following instructions: {schema}\n\n"
            "You decide the question is: "
        )

    def evaluate(self, dataloader, return_f2=False):
        self.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["label"].to(self.model.device).view(-1, 1).float()

                logits = self.forward(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f2, _ = precision_recall_fscore_support(all_labels, all_preds, beta=2, average="binary")

        logger.info(f"Validation — Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F2: {f2:.3f}")
        return f2 if return_f2 else None


def load_abstention_classifier_embed(path: str = ".local/AbstentionClassifier/binary_head/optuna_trial_n14_f2_0.833_epoch0.pt"):
    device = torch.device("cuda")
    model = AbstentionClassifierEmbeddingBased(frozen=True)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


class AbstentionClassifierEmbeddingBased(AbstentionClassifier):
    def __init__(self, frozen=True, dropout_rate=0.3):
        super().__init__(frozen)
        self.classifier = nn.Linear(self.hidden_size * 3, 1).to(self.device)
        self.dropout = nn.Dropout(p=dropout_rate)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.prompt_template = (
            "You are a data scientist who has to create SQL queries for users, but the users do not know what content the database has.\n"
            "You have to read over a users question, and compare it to the database schema that you recieve, and then decide if the users question can be answered based on the database.\n"
            "You have received the question: \"{question}\" \n"
            "The database that the user is asking the question of is described by the following schema: \n {schema}\n\n"
            "After looking over the schema very carefully, and making sure to catch all questions that cannot be answered.\n"
            "Is the question {question} answerable by the schema? [yes] or [no]"
        )

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        logits = []

        tokens_batch = [self.tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]

        for i in range(batch_size):
            tokens = tokens_batch[i]

            def find_token_span(tokens, target_span):
                for idx in range(len(tokens) - len(target_span) + 1):
                    if tokens[idx:idx+len(target_span)] == target_span:
                        return idx
                return -1

            yes_span = ['Ġ[', 'yes', ']']
            no_span = ['Ġ[', 'no', ']']

            yes_start = find_token_span(tokens, yes_span)
            no_start = find_token_span(tokens, no_span)

            if yes_start == -1 or no_start == -1:
                logger.error("Missing [yes] or [no] in token sequence")
                raise ValueError("Missing [yes] or [no] token span.")

            yes_emb = hidden_states[i, yes_start + 1, :]
            no_emb = hidden_states[i, no_start + 1, :]

            combined_emb = torch.cat([no_emb, yes_emb, no_emb - yes_emb], dim=-1).to(dtype=self.model.dtype)

            combined_emb = self.dropout(combined_emb)

            logit = self.classifier(combined_emb)

            logits.append(logit)

        logits = torch.stack(logits).squeeze(-1)
        return logits

    def classify(self, user_question, schema, threshold=0.5):
        self.eval()
        prompt = self.prompt_template.format(question=user_question, schema=schema)

        inputs = self.tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            logit = self.forward(inputs['input_ids'], inputs['attention_mask'])
            prob = torch.sigmoid(logit).squeeze().item()

        return "feasible" if prob > threshold else "infeasible"

    def fine_tune(self, data, epochs=3, lr=1e-4, batch_size=8, val_split=0.1, weight_decay=0.01, save_path=HEAD_SAVE_LOCALE):
        dataset = SQLFeasibilityDataset(
            data,
            8192,
            self.prompt_template,
            MODEL_NAME,
            self.model.dtype
        )

        torch.manual_seed(42)
        random.seed(42)

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=NUM_WORKERS)

        optimizer = optim.AdamW(
            list(filter(lambda p: p.requires_grad, self.parameters())),
            lr=lr,
            weight_decay=weight_decay
        )

        loss_func = nn.BCEWithLogitsLoss()
        best_f2 = 0.0

        logger.info(f'Training model for {epochs} epochs...')
        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1).float()

                logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

            f2 = self.evaluate(val_loader, return_f2=True)

            if f2 > best_f2:
                best_f2 = f2
                model_path = save_path + f'_f2_{best_f2:.3f}_epoch{epoch}.pt'
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.state_dict(), model_path)
                logger.info(f"New best F2: {best_f2:.4f} — Model saved to: {model_path}")

        return best_f2


class SQLFeasibilityDataset(Dataset):
    def __init__(self, data, max_length, prompt_template, model=MODEL_NAME, dtype=torch.float):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = max_length
        self.dtype = dtype
        self.prompt_template = prompt_template
        self.data = self._filter_examples(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        prompt = self.prompt_template.format(question=sample['question'], schema=sample['schema'])
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        label = torch.tensor(sample['feasible'], dtype=self.dtype)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

    def _filter_examples(self, data):
        filtered = []
        removed = 0
        for sample in data:
            prompt = (
                "You are a data scientist, who has to vet questions from users.\n"
                f"You have received the question: \"{sample['question']}\" "
                f"for the database described by the following instructions: {sample['schema']}\n\n"
                "You decide the question is: "
            )
            tokens = self.tokenizer(prompt)
            if len(tokens['input_ids']) <= self.max_length:
                filtered.append(sample)
            else:
                removed += 1

        logger.info(f"Dataset filtering complete, removed {removed} examples.")
    
    def train_val_split(self, val_fraction=0.1, seed=33):
        db_id_to_samples = defaultdict(list)

        for sample in self.data:
            db_id_to_samples[sample['db_id']].append(sample)
        
        db_ids = list(db_id_to_samples.keys())
        random.seed(seed)
        random.shuffle(db_ids)

        total_samples = len(self.data)
        target_val_samples = int(total_samples * val_fraction)

        val_data = []
        train_data = []
        cum_val = 0

        val_db_ids = set()

        for db_id in db_ids:
            db_samples = db_id_to_samples[db_id]
            if cum_val < target_val_samples:
                val_data.extend(db_samples)
                cum_val += len(db_samples)
                val_db_ids.add(db_id)
            else:
                train_data.extend(db_samples)
        logger.info(
            f"Train/Eval split complete — Train Samples: {len(train_data)}, Eval Samples: {len(val_data)}, "
            f"Eval DBs: {len(val_db_ids)}, Total Samples: {total_samples}"
        )
        return (
            SQLFeasibilityDataset(train_data, self.max_length, self.prompt_template, model=self.tokenizer.name_or_path, dtype=self.dtype),
            SQLFeasibilityDataset(val_data, self.max_length, self.prompt_template, model=self.tokenizer.name_or_path, dtype=self.dtype)
        )