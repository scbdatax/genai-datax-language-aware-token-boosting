import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from latb.llms import BaseHFLLM

class SimpleXLSUMGenEngine:
    """A simple data engine for xlsum"""
    def __init__(
            self, 
            model: BaseHFLLM,
            lang: str,
            data_path: str,
            data_split: str = 'test',
        ):
        self.responses = {
            "id": [],
            "gt": [],
            "pred": []
        }

        # dataset
        self.ds = load_dataset(
            data_path, 
            lang,
            trust_remote_code=True
        )[data_split]
        
        # subsample
        if len(self.ds) > 1_000:
            self.ds = self.ds.select(range(1_000))

        # model
        self.model = model

    def generate(self, save_path: str) -> None:
        for d in tqdm(self.ds):
            # extract data
            id_ = d["id"]
            text = d["text"]
            summary = d["summary"]

            # generate response
            response = self.model.gen_response(text)

            # update response
            self.responses["id"].append(id_)
            self.responses["gt"].append(summary)
            self.responses["pred"].append(response)

            # save response
            pd.DataFrame(self.responses).to_csv(save_path, index=False)