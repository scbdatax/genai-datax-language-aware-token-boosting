from transformers import AutoTokenizer

from latb.evaluate import BaseEvaluator


class AverageTokenLength(BaseEvaluator):
    def __init__(
            self,
            tokenizer_path: str
        ):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def eval(
            self, 
            file_path: str, # csv file
        ):
        '''
        Compute average token length for all responses
        '''
        
        assert "csv" in file_path, "file_path must be a csv file."
        
        self._load_data(file_path=file_path)

        token_lengths = []
        for p in self.predictions:
            token_lengths.append(len(self.tokenizer.tokenize(p)))

        return sum(token_lengths) / len(token_lengths)


if __name__ == '__main__':

    import os
    import argparse
    from dotenv import load_dotenv
    from transformers import AutoTokenizer
    
    parser = argparse.ArgumentParser(
        description='Please provide the file path and target language id'
    )

    parser.add_argument('-f', '--file', required=True)
    args = vars(parser.parse_args())
    file_path = args['file']

    # paths
    load_dotenv(dotenv_path="../../.env")
    tokenizer_path = os.getenv("LLAMA3_PATH")

    avg_tok_len = AverageTokenLength(tokenizer_path=tokenizer_path)

    res = avg_tok_len.eval(file_path=file_path)
    print("Average Token Length: {:.2f}".format(res))