import os
from typing import List, Dict
from dotenv import load_dotenv

from latb.llms import BaseHFLLM
from transformers import GenerationConfig
from experiments.sum.baselines.simple_xlsum_data_engine import SimpleXLSUMGenEngine

from latb.utils.lang import xlsum_prompt_template


class SuzumeXLSum(BaseHFLLM):
    def __init__(
            self, 
            model_path: str, 
            use_flash_attn_2: bool,
            generation_config: GenerationConfig
        ):
        super().__init__(
            model_path=model_path,
            use_flash_attn_2=use_flash_attn_2,
            generation_config=generation_config
        )
        self.lang_code = None
    
    def set_lang_code(self, lang_code: str):
        self.lang_code = lang_code
    
    def _build_template(self, text: str) -> List[Dict]:
        assert self.lang_code != None, "Please set the lang_code attribute"
        template = [
            {
                "role": "system", 
                "content": "You are a multilingual writer which summarize text very well. Write a short summary."
            },
            {
                "role": "user",
                "content": xlsum_prompt_template[self.lang_code].format(text)
            }
        ]
        return template

if __name__ == "__main__":
    # load env
    load_dotenv(dotenv_path="../../../.env")

    # generation config
    generation_config = GenerationConfig(
        max_new_tokens=1_000,
        temperatue=1,
        top_p=1,
        do_sample=True,
        num_return_sequences=1
    )

    # model
    suzume_path = os.getenv("SUZUME_PATH")
    model = SuzumeXLSum(
        model_path=suzume_path, 
        use_flash_attn_2=False,
        generation_config=generation_config
    )

    # languages
    langs = {
        "lang_name": ["french", "japanese", "russian", "chinese_simplified"],
        "lang_code": ["en", "ru", "zh", "ja", "fr", "th", "hi", "ko", "ar"]
    }

    # loop through languages
    for i in range(len(langs["lang_name"])):
        lang_name, lang_code = langs["lang_name"][i], langs["lang_code"][i]

        # data engine
        data_path = os.getenv("DATA_PATH")
        data_engine = SimpleXLSUMGenEngine(model=model, lang=lang_name, data_path=data_path)

        # set language for model
        model.set_lang_code(lang_code=lang_code)

        # generate
        data_engine.generate(save_path=f"xlsum-{lang_name}_suzume.csv")