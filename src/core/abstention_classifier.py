from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from src.common.logger import get_logger
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import random
import os

MODEL_NAME = "XGenerationLab/XiYanSQL-QwenCoder-7B-2502"
HEAD_SAVE_LOCALE = '.local/AbstentionClassifier/BinaryHead/best_classifier.pt'
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

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state

        pooled = last_hidden[:, -1, :]
        logit = self.classifier(pooled)
        return logit

    def classify(self, user_question, schema, threshold=0.5):
        prompt = self.prompt_template.format(question=user_question, schema=schema)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            logit = self.forward(inputs['input_ids'].to(self.model.device), attention_mask)
            prob = torch.sigmoid(logit).item()

        return 'feasible' if prob > threshold else 'infeasible'

    def load_classifier(self, classifier_locale=HEAD_SAVE_LOCALE):
        state_dict = torch.load(classifier_locale, map_location=self.model.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.to(self.model.device)

    def fine_tune(self, data, epochs=3, lr=1e-4, batch_size=8, val_split=0.1, weight_decay=0.01, save_path=HEAD_SAVE_LOCALE):
        dataset = SQLFeasibilityDataset(data, 8192, prompt_template=self.prompt_template, model=MODEL_NAME, dtype=self.model.dtype)
        torch.manual_seed(42)
        random.seed(42)

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        optimizer = optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        loss_func = nn.BCEWithLogitsLoss()

        best_f2 = 0.0

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in tqdm(train_loader):
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["label"].to(self.model.device).unsqueeze(1)

                logits = self.forward(input_ids, attention_mask)
                loss = loss_func(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
            f2 = self.evaluate(val_loader, True)

            if f2 > best_f2:
                best_f2 = f2
                os.makedirs(os.path.dirname(HEAD_SAVE_LOCALE), exist_ok=True)
                torch.save(self.state_dict(), HEAD_SAVE_LOCALE)
                logger.info(f"Saved best classifier with F2: {best_f2:.4f}")
        return best_f2

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


def load_abstention_classifier(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AbstentionClassifier(frozen=True)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


class AbstentionClassifierEmbeddingBased(AbstentionClassifier):
    def __init__(self, frozen=True):
        super().__init__(frozen)
        self.yes_token, self.no_token = '<<yes>>', '<<no>>'
        self.classifier = nn.Linear(self.hidden_size * 2, 1).to(self.device)
        self.prompt_template = (
            "You are a data scientist, who has to vet questions from users.\n"
            "You have received the question: \"{question}\" "
            "for the database described by the following instructions: {schema}\n\n"
            "You decide the question is: <<yes>> or <<no>>"
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        logits = []

        tokens = [self.tokenizer.convert_ids_to_tokens(seq) for seq in input_ids]

        for i in range(batch_size):
            try:
                yes_idx = tokens[i].index(self.yes_token)
                no_idx = tokens[i].index(self.no_token)
            except ValueError:
                logger.error(f"Missing <<yes>> or <<no>> token in input sample.")

            yes_emb = hidden_states[i, yes_idx, :]
            no_emb = hidden_states[i, no_idx, :]
            combined = torch.cat([yes_emb, no_emb], dim=1)
            logit = self.classifier(combined)
            logits.append(logit)

        return torch.stack(logits)

    def classify(self, user_question, schema, threshold=0.5):
        prompt = self.prompt_template.format(question=user_question, schema=schema)

        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logit = self.forward(input_ids, attention_mask)
            prob = torch.sigmoid(logit).item()

        return "feasible" if prob > threshold else "infeasible"


class SQLFeasibilityDataset(Dataset):
    def __init__(self, data, max_length, prompt_template, model=MODEL_NAME, dtype=torch.float):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.data = data
        self.max_length = max_length
        self.dtype = dtype
        self.prompt_template = prompt_template

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
