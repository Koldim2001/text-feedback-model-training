import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ---------------------- Ваш класс модели (без изменений) ----------------------
class E5Classifier(nn.Module):
    def __init__(self, model_name='intfloat/multilingual-e5-small'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        print("embed_dim = ", self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1)   # -> [batch]

# ---------------------- Загрузка весов и экспорт ----------------------
def export_onnx(checkpoint_path, output_path="model.onnx", max_length=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Создаём модель (такой же архитектуры, как при обучении)
    model = E5Classifier(model_name='intfloat/multilingual-e5-small')
    model.to(device)

    # 2. Загружаем state_dict из .pth
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')

    # 4. Обёртка, чтобы добавить размерность [batch] -> [batch, 1]
    class ModelWithReshape(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model

        def forward(self, input_ids, attention_mask):
            out = self.original_model(input_ids, attention_mask)  # [batch]
            return out.unsqueeze(-1)  # [batch, 1]

    wrapped_model = ModelWithReshape(model)
    wrapped_model.eval()

    # 5. Подготовка фиктивного входа
    dummy_text = "dummy text for onnx export"
    dummy_inputs = tokenizer(
        dummy_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    dummy_input_ids = dummy_inputs['input_ids'].to(device)
    dummy_attention_mask = dummy_inputs['attention_mask'].to(device)

    # 6. Экспорт
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    print(f"Модель успешно экспортирована в {output_path} с выходом [batch, 1]")

# ---------------------- Запуск ----------------------
if __name__ == "__main__":
    export_onnx(
        checkpoint_path="best_model.pth",   # путь к вашему .pth
        output_path="model.onnx",
        max_length=64
    )
