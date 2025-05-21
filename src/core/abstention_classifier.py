from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import torch

MODEL_NAME = "XGenerationLab/XiYanSQL-QwenCoder-7B-2502"


class AbstentionClassifier():
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            torch_dtype='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.allowed_tokens = ['answerable', 'unanswerable']
        self.max_tokens = 10
        self.allowed_token_sequences = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in self.allowed_tokens
        ]
        self.prefix_tree = self._build_prefix_tree(self.allowed_token_sequences)
        self.logits_processor = PrefixConstrainedLogitsProcessor(self.prefix_tree)
        self.fallback_answer = "unanswerable"

    def classify(self, user_question, schema):
        prompt = self._fit_prompt(user_question, schema)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
        generated_ids = input_ids[0].tolist()
        generated_tokens = []

        for _ in range(self.max_tokens):
            input_tensor = torch.tensor([generated_ids]).to(self.model.device)
            outputs = self.model(input_ids=input_tensor)
            logits = outputs.logits[:, -1, :]
            scores = logits.squeeze(0)

            self.logits_processor.generated_tokens = generated_tokens
            scores = self.logits_processor(input_tensor, scores)

            next_token_id = torch.argmax(scores).item()
            generated_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            for label_seq in self.allowed_token_sequences:
                if generated_tokens == label_seq:
                    return self.tokenizer.decode(label_seq)

        return self.fallback_answer

    def _fit_prompt(self, user_question: str, schema: str):
        return (
            "You are a data scientist, who has to vet questions from users.\n"
            f"You have received the question: \"{user_question}\" "
            f"for the database described by the following instructions: {schema}\n\n"
            "You decide the question is: "
        )

    def _build_prefix_tree(self, sequences):
        tree = {}
        for seq in sequences:
            node = tree
            for token_id in seq:
                node = node.setdefault(token_id, {})
            node[None] = None
        return tree


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_tree):
        super().__init__()
        self.prefix_tree = prefix_tree
        self.current_node = prefix_tree
        self.generated_tokens = []

    def __call__(self, input_ids, scores):
        if not self.generated_tokens:
            self.current_node = self.prefix_tree    
        else:
            for token_id in self.generated_tokens:
                self.current_node = self.current_node.get(token_id, {})
                if self.current_node is None:
                    break

        if self.current_node is None:
            scores[:] = -float('inf')
            return scores

        allowed_next_tokens = list(self.current_node.keys())
        if None in allowed_next_tokens:
            allowed_next_tokens.remove(None)

        mask = torch.full_like(scores, -float('inf'))
        mask[allowed_next_tokens] = 0
        scores = scores + mask
        return scores
