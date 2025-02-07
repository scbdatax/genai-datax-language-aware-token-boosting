import os

import argparse
import pandas as pd

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, GenerationConfig

from typing import Optional

from latb.lang_id import LanguageID
from latb.utils.lang import xlsum_lang_map, xlsum_prompt_template
from latb.logit_processors import BoostLogitsScoreProcessor, AdaptiveBoostLogitsScoreProcessor


class XLSUM_Gen_Response:
    def __init__(
            self, 
            model_path: str, 
            data_path: str,
            output_path: str,
            lang_code: str,
            gen_config: GenerationConfig,
            use_latb: bool,
            adaptive: bool,
            thresh_diff: float,
            latb_value: float = 0,
            lid_weight_path: Optional[str]=None,
            token_lang_id_method: str = 'unicode',
            data_split: str = 'test',
            subsample_data: int = -1 # for testing with less data, -1 for disable
        ):
        self.tokenizer = None
        self.model = None
        self.terminators = None
        self.data = None
        self.responses = {
            "id": [],
            "gt": [],
            "pred": []
        }

        # make sure we use the language in the dataset
        assert lang_code in xlsum_lang_map.keys()

        self.lang_code = lang_code
        self.lang = xlsum_lang_map[lang_code]
        self.output_path = output_path
        self.use_latb = use_latb
        self.adaptive = adaptive
        self.thresh_diff = thresh_diff
        self.latb_value = latb_value
        self.generation_config = gen_config
        self.data_split = data_split
        self.subsample_data = subsample_data
        self.model_path = model_path
        self.lid_weight_path = lid_weight_path
        self.data_path = data_path

        self.target_lang_token_ids = []
        self.num_token_ids = []
        self.sp_token_ids = []

        self._load_model()
        if use_latb:
            vocab = self.tokenizer.get_vocab()
            self.ids_tokens_map = {
                v: self.tokenizer.convert_tokens_to_string([k]) \
                    for k, v in vocab.items()
            }
            # target language tokens
            if token_lang_id_method == 'pyThaiNLP':
                self.target_lang_token_ids = self._detect_token_thai()
            elif token_lang_id_method == 'fastText':
                self.target_lang_token_ids = self._detect_token_lang_fasttext()
            elif token_lang_id_method == 'unicode':
                self.target_lang_token_ids = self._detect_token_lang_unicode(self.lang_code)
            else:
                raise NotImplementedError
            # numbers
            self.num_token_ids = self._detect_token_lang_unicode("num")
            # special characters
            self.sp_token_ids = self._detect_token_lang_unicode("sp")
        
        self._load_data()

    def _load_model(self):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "left"

        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda"
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # terminators
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def _detect_token_lang_unicode(self, lang: str) -> list:
        token_list = []
        lid = LanguageID()
        # language identification
        for id, token in self.ids_tokens_map.items():
            is_in_lang = lid.is_in_lang(token, lang)
            if (is_in_lang):
                token_list.append(id)
        return token_list

    def _detect_token_lang_fasttext(self) -> list:
        token_list = []
        lid = LanguageID(self.lid_weight_path)
        # language identification
        for id, token in self.ids_tokens_map.items():
            pred_lang = lid.predict(token)[0]
            if (pred_lang == self.lang_code):
                token_list.append(id)
        return token_list
    
    def _detect_token_thai(self) -> list:
        token_list = []
        lid = LanguageID()
        # language identification
        for id, token in self.ids_tokens_map.items():
            is_thai = lid.is_thai(token)
            if (is_thai):
                token_list.append(id)
        return token_list

    def _load_data(self):
        ds = load_dataset(
            self.data_path, 
            self.lang,
            trust_remote_code=True
        )
        print(f"Data length before subsample: {len(ds[self.data_split])}")
        if (self.subsample_data == -1) or (self.subsample_data >= len(ds[self.data_split])):
            self.data = ds[self.data_split]
        else:
            self.data = ds[self.data_split].select(range(self.subsample_data))
        print(f"Data length after subsample: {len(self.data)}")

    def _save_res(self, output_path: str):
        if self.use_latb:
            latb_text = "true"
        else:
            latb_text = "false"
        save_path = os.path.join(output_path, f"xlsum-{self.lang}_latb-{latb_text}.csv")
        pd.DataFrame(self.responses).to_csv(save_path, index=False)
    
    def _prompt_generator(self, query: str) -> str:
        return xlsum_prompt_template[self.lang_code].format(query)

    def generate(
            self, 
            save_freq: int = 5, 
            batch_size: int = 10
        ):
        if self.use_latb:
            if self.adaptive:
                latb = AdaptiveBoostLogitsScoreProcessor(
                    target_lang_token_ids=self.target_lang_token_ids,
                    num_token_ids=self.num_token_ids,
                    sp_token_ids=self.sp_token_ids,
                    thresh_diff=self.thresh_diff,
                    boost_value=self.latb_value
                )
            else:
                latb = BoostLogitsScoreProcessor(
                    target_lang_token_ids=self.target_lang_token_ids,
                    num_token_ids=self.num_token_ids,
                    sp_token_ids=self.sp_token_ids,
                    boost_value=self.latb_value
                )
            logits_processors = LogitsProcessorList([latb])
        else:
            logits_processors = LogitsProcessorList([])

        batched_data = self.data.batch(batch_size=batch_size)
        
        count = 0
        for d in tqdm(batched_data):
            id_ = d["id"]
            text = d["text"]
            summary = d["summary"]

            input_texts = []
            for i in range(len(text)):
                tmp = [
                    {"role": "system", "content": "You are a multilingual writer which summarize text very well. Write a short summary."}
                ]
                tmp.append({
                    "role": "user",
                    "content": self._prompt_generator(text[i])
                })
                input_texts.append(tmp)

            input_ids = self.tokenizer.apply_chat_template(
                input_texts, padding=True, tokenize=True, 
                add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            # input_ids = inputs['input_ids']
            outputs = self.model.generate(
                inputs=input_ids,
                generation_config=self.generation_config,
                logits_processor=logits_processors,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id
            )[:, input_ids.shape[1]:]
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i in range(len(id_)):
                self.responses["id"].append(id_[i])
                self.responses["gt"].append(summary[i])
                self.responses["pred"].append(output_texts[i])

            if count % save_freq == 0:
                self._save_res(self.output_path)
            
            count += batch_size
        
        self._save_res(self.output_path)

if __name__ == '__main__':

    def str2bool(x):
        if x == "true":
            return True
        elif x == "false":
            return False
        else:
            raise argparse.ArgumentTypeError('true or false expected.')
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_code', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--use_latb', type=str2bool, default=True)
    parser.add_argument('--adaptive', type=str2bool, default=True)
    parser.add_argument('--thresh_diff', type=float, default=0.80)
    parser.add_argument('--token_lang_id_method', type=str, default='unicode')
    parser.add_argument('--latb_value', type=float, default=50)
    parser.add_argument('--subsample_data', type=int, default=-1)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--temperatue', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--do_sample', type=str2bool, default=True)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    args = vars(parser.parse_args())

    gen_config = GenerationConfig(
        max_new_tokens=args['max_new_tokens'],
        temperatue=args['temperatue'],
        top_p=args['top_p'],
        do_sample=args['do_sample'],
        num_return_sequences=args['num_return_sequences']
    )
    data_engine = XLSUM_Gen_Response(
        lang_code=args['lang_code'],
        model_path=args['model_path'],
        data_path=args['data_path'],
        output_path=args['output_path'],
        use_latb=args['use_latb'],
        adaptive=args['adaptive'],
        thresh_diff=args['thresh_diff'],
        token_lang_id_method=args['token_lang_id_method'],
        latb_value=args['latb_value'],
        subsample_data=args['subsample_data'],
        gen_config=gen_config
    )

    data_engine.generate(
        save_freq=args['save_freq'],
        batch_size=args['batch_size']
    )