from typing import List, Dict
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class BaseHFLLM(ABC):
    """Abstract class for HuggingFace Transformers LLM call"""
    def __init__(
            self, 
            model_path: str, 
            use_flash_attn_2: bool,
            generation_config: GenerationConfig
        ):
        self.model_path = model_path
        self.use_flash_attn_2 = use_flash_attn_2
        self.generation_config = generation_config

        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load Hugging Face model"""
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # model
        if self.use_flash_attn_2:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="cuda"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16,
                device_map="cuda"
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # terminators
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    @abstractmethod
    def _build_template(self, text: str) -> List[Dict]:
        """Build a chat template from query text"""
        pass

    def gen_response(self, text: str) -> str:
        """
        Generate a response from the LLM based on query text
        Args:
            text (str): Input text for the LLM
        Returns:
            A generated response based on query text (str)
        """
        # build template
        text_with_template = self._build_template(text)

        # apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            text_with_template, padding=True, tokenize=True, 
            add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # generate response
        response_ids = self.model.generate(
            inputs=input_ids,
            generation_config=self.generation_config,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )[0, input_ids.shape[1]:]

        # decode output
        response = self.tokenizer.decode(
            response_ids, 
            skip_special_tokens=True
        )

        return response

