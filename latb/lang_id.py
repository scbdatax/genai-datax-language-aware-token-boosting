import re
import regex
import fasttext
import pythainlp

from typing import List, Tuple, Optional

from latb.utils.lang import is_char_in_lang


class LanguageID:
    def __init__(self, weight_path: Optional[str]=None) -> None:
        if weight_path != None: 
            self.model = fasttext.load_model(weight_path)

    def _clean_text(self, text: str):
        cleaned_text = regex.sub(r'\d+', '', text) # remove numbers
        cleaned_text = regex.sub(r'\p{P}+', '', cleaned_text) # remove punctuations
        cleaned_text = regex.sub(r'[^\p{L}\p{M}\s]', '', cleaned_text) # remove special characters
        cleaned_text = regex.sub('\n', '', cleaned_text) # remove new line character
        cleaned_text = regex.sub(' ', '', cleaned_text) # remove space
        return cleaned_text

    def predict(
        self, 
        text: str,
        k: int = 1,
    ) -> List[str]:
        """
        Identify the language of the text using fasttext library.

        Args:
            text (str): text that we want to identify the language
            k (int): number of tokens that

        Returns:
            Detected language is returned in ISO code.
        """

        # make sure that we've loaded the fasttext model
        assert self.model != None

        text = self._clean_text(text)

        if text == '':
            return ['']
        
        predictions = self.model.predict(text, k=k)[0]
        langs = []
        for p in predictions:
            langs.append(p.replace('__label__', ''))

        return langs

    def predict_with_prob(
        self, 
        text: str,
        k: int = 1,
    ) -> Tuple[List[str], List[float]]:
        """
        Identify the language of the text using fasttext library.

        Args:
            text (str): text that we want to identify the language
            k (int): number of tokens that

        Returns:
            Tuple of detected language and probability of confidence
        """

        # make sure that we've loaded the fasttext model
        assert self.model != None
        
        text = self._clean_text(text)

        if text == '':
            return ['']
        
        predictions = self.model.predict(text, k=k)
        langs = []
        probs = predictions[1]
        for p in predictions[0]:
            langs.append(p.replace('__label__', ''))

        return langs, probs

    def is_thai(self, text: str) -> bool:
        """
        Return whether the given text is in Thai language

        Args:
            text (str): text that we want to identify

        Returns:
            boolean determining if the given text is Thai 
        """

        text = self._clean_text(text)

        if text == '':
            return False
        else:
            return pythainlp.util.isthai(text)
    
    def is_in_lang(self, text: str, lang: str) -> bool:
        """
        Return whether the given text is in a given language

        Args:
            text (str): text that we want to identify
            lang (str): language code

        Returns:
            boolean determining if the given text is in a given language 
        """
        # filter the unknown character
        if '�' in text:
            return False
        
        is_lang = is_char_in_lang[lang]

        if (lang != "num") and (lang != "sp"):
            text = self._clean_text(text)

        if text == '':
            return False
        else:
            for char in text:
                if not is_lang(char):
                    return False
            return True
