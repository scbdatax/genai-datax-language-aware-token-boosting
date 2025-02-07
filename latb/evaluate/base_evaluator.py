import pandas as pd


class BaseEvaluator:
    eval_data = None
    predictions = None
    
    def _load_data(self, file_path: str):
        self.eval_data = pd.read_csv(file_path)
        self.predictions = self.eval_data["pred"].to_list()
        
    def eval(self, file_path: str, target_lang_id: str):
        raise NotImplementedError
