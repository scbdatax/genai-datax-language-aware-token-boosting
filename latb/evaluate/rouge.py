import evaluate
import pandas as pd

from typing import Callable

from latb.evaluate import BaseEvaluator


class RougeEval(BaseEvaluator):
    def __init__(
            self,
            tokenizer_fn: Callable, # tokenizer function
        ):
        # rouge
        self.rouge = evaluate.load('rouge')
        # tokenizer
        self.tokenizer_fn = tokenizer_fn
        # eval data
        self.references = None
    
    def _load_data(self, file_path: str):
        self.eval_data = pd.read_csv(file_path)
        self.predictions = self.eval_data["pred"].to_list()
        self.references = self.eval_data["gt"].to_list()

    def eval(
            self, 
            file_path: str # csv file
        ):
        '''
        Compute Rouge-1, Rouge-2, and Rouge-L between
        predictions and references
        '''
        
        self._load_data(file_path=file_path)
        ret = self.rouge.compute(
            predictions=self.predictions,
            references=self.references,
            tokenizer=self.tokenizer_fn
        )
        return ret


if __name__ == '__main__':

    import os
    import argparse
    from dotenv import load_dotenv
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(
        description='Please provide the file path'
    )

    parser.add_argument('-f', '--file', required=True)
    args = vars(parser.parse_args())
    file_path = args['file']

    # load tokenizer
    load_dotenv(dotenv_path="../../.env")
    local_model_path = os.getenv("LLAMA3_PATH")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer_fn = tokenizer.tokenize

    rouge_eval = RougeEval(
        tokenizer_fn=tokenizer_fn
    )

    res = rouge_eval.eval(file_path=file_path)
    print(res)