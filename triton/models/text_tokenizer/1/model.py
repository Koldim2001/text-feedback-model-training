import os
import numpy as np
import transformers
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_tokenizer_dir = os.path.join(model_dir, "tokenizer")
        
        # Используем новую модель
        model_name = "intfloat/multilingual-e5-small"

        # Try local tokenizer first (recommended for offline inference),
        # fallback to hub if not present.
        if os.path.isdir(local_tokenizer_dir):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                local_tokenizer_dir, local_files_only=True
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name
            )
        
        self.max_length = 64

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            # Получаем входной текст
            raw_texts = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = [t.decode("utf-8") for t in raw_texts.flatten()]

            # Токенизация с параметрами, совместимыми с ONNX экспортом
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            
            # Подготовка тензоров
            # E5 и ONNX обычно требуют input_ids и attention_mask
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

            out_tensors = [
                pb_utils.Tensor("input_ids", input_ids),
                pb_utils.Tensor("attention_mask", attention_mask)
            ]
            
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses