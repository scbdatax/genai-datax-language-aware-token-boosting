import regex

from transformers import AutoTokenizer

from latb.lang_id import LanguageID
from latb.evaluate import BaseEvaluator


class LangConfusion(BaseEvaluator):
    def __init__(
            self,
            tokenizer_path: str,
            language_id_path: str
        ):
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # language identification
        self.lid = LanguageID(language_id_path)
    
    def _clean_text(self, text: str):
        cleaned_text = regex.sub(r'\d+', '', text) # remove numbers
        cleaned_text = regex.sub(r'\p{P}+', '', cleaned_text) # remove punctuations
        cleaned_text = regex.sub(r'[^\p{L}\p{M}\s]', '', cleaned_text) # remove special characters
        cleaned_text = regex.sub('\n', '', cleaned_text) # remove new line character
        return cleaned_text

    def eval_token_level_unicode(
            self, 
            file_path: str, # csv file
            target_lang_id: str
        ):
        '''
        Compute average undesire token unicode output rate (%)
        '''
        self._load_data(file_path=file_path)

        negative_samples = []
        confusion_rates = []
        for p in self.predictions:
            tokens = [
                self.tokenizer.convert_tokens_to_string([x]) 
                for x in self.tokenizer.tokenize(p)
            ]
            counter = 0
            for t in tokens:
                is_in_lang = self.lid.is_in_lang(t, target_lang_id)
                is_num = self.lid.is_in_lang(t, "num")
                is_sp = self.lid.is_in_lang(t, "sp")
                if (not is_in_lang) and (not is_num) and (not is_sp):
                    counter += 1
                    negative_samples.append(t)
            confusion_rates.append(counter / len(tokens))
        return sum(confusion_rates) / len(confusion_rates), negative_samples
    
    def eval_line_level_fasttext(
            self, 
            file_path: str, # csv file
            target_lang_id: str
        ):
        '''
        Compute average language confusion at line level (%)
        '''
        self._load_data(file_path=file_path)

        negative_samples = []
        line_level_correct_rates = []
        for p in self.predictions:
            lines = p.split('\n')
            confused_line = 0
            num_lines = 0
            for l in lines:
                l = self._clean_text(l)
                if l == '':
                    continue
                lang_id = self.lid.predict(l)[0]
                if lang_id != target_lang_id:
                    confused_line += 1
                    negative_samples.append(l)
                num_lines += 1
            line_level_correct_rates.append(confused_line / num_lines)
                
        return sum(line_level_correct_rates) / len(line_level_correct_rates), negative_samples
    
    def eval_response_level_fasttext(
            self, 
            file_path: str, # csv file
            target_lang_id: str
        ):
        '''
        Compute average language confusion at response level (%)
        '''
        self._load_data(file_path=file_path)

        negative_samples = []
        confusion_count = 0
        for p in self.predictions:
            p = self._clean_text(p)
            lang_id = self.lid.predict(p)[0]
            if lang_id != target_lang_id:
                confusion_count += 1
                negative_samples.append(p)
        return confusion_count / len(self.predictions), negative_samples


if __name__ == '__main__':

    import os
    import argparse
    from dotenv import load_dotenv
    from transformers import AutoTokenizer

    
    parser = argparse.ArgumentParser(
        description='Please provide the file path and target language id'
    )

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-l', '--lang', required=True)
    args = vars(parser.parse_args())
    file_path = args['file']
    target_lang_id = args['lang']

    # paths
    load_dotenv(dotenv_path="../../.env")
    tokenizer_path = os.getenv("LLAMA3_PATH")
    language_id_path = os.getenv("FASTTEXT_PATH")

    lang_confusion = LangConfusion(
        tokenizer_path=tokenizer_path,
        language_id_path=language_id_path
    )

    res_token_level, _ = lang_confusion.eval_token_level_unicode(
        file_path=file_path, 
        target_lang_id=target_lang_id
    )

    print("Token-level Language Confusion Rate: {:.2f} %".format(res_token_level*100))

    res_line_level, _ = lang_confusion.eval_line_level_fasttext(
        file_path=file_path, 
        target_lang_id=target_lang_id
    )

    print("Line-level Language Confusion Rate: {:.2f} %".format(res_line_level*100))

    res_response_level, _ = lang_confusion.eval_response_level_fasttext(
        file_path=file_path, 
        target_lang_id=target_lang_id
    )

    print("Response-level Language Confusion Rate: {:.2f} %".format(res_response_level*100))