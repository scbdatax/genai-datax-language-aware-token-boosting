import torch

from transformers import LogitsProcessor


class BoostLogitsScoreProcessor(LogitsProcessor):
    """
    Simply add a value to the specified logit_ids

    Args:
        boost_token_ids (list): List of token ids that will be boosted
        boost_value (float): The value that will be added directly to the logits of 
            specified tokens in the boost_token_ids
    """
    def __init__(
            self, 
            target_lang_token_ids: list,
            num_token_ids: list,
            sp_token_ids: list,
            boost_value: float,
            eos_id: int = 128009 # for llama3
        ):
        self.target_lang_token_ids = target_lang_token_ids
        self.num_token_ids = num_token_ids
        self.sp_token_ids = sp_token_ids
        self.boost_value = boost_value
        self.eos_id = eos_id

        self.boost_token_ids = None
        self._set_boost_token_ids()
    
    def _set_boost_token_ids(self):
        """Boost target language characters, numbers, and special characters"""
        self.boost_token_ids = self.target_lang_token_ids + self.num_token_ids + self.sp_token_ids
        self.boost_token_ids.append(self.eos_id)

    def __call__(
            self, 
            input_ids: torch.LongTensor, 
            scores: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Runs when the class is called

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length)): 
                Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor of shape (batch_size, config.vocab_size)): 
                Prediction scores of a language modeling head. 
                These can be logits for each vocabulary when not using beam search 
                or log softmax for each vocabulary token when using beam search
        
        Returns:
            torch.FloatTensor of shape (batch_size, config.vocab_size)
        """
        scores[..., self.boost_token_ids] += self.boost_value
        return scores

class AdaptiveBoostLogitsScoreProcessor(LogitsProcessor):
    """
    Add a value to the target language logit_ids only if the probability is flat
    We divide token into 2 categories
    1. target language tokens
    2. other tokens
    If the difference of the probability of two categories is less than thresh_diff,
    We then add the boost_value to the target_lang_token_ids, otherwise, we do nothing.

    Args:
        target_lang_token_ids (list): List of token ids that is in the target language
        num_token_ids (list): List of number token ids
        sp_token_ids (list): List of special characters token ids
        thresh_diff (float): Threshold in probability difference (>=0, <=1) determining if the probability if flat.
        boost_value (float): The value that will be added directly to the logits of 
            specified tokens in the boost_token_ids
    """
    def __init__(
            self, 
            target_lang_token_ids: list,
            num_token_ids: list,
            sp_token_ids: list,
            thresh_diff: float,
            boost_value: float,
            eos_id: int = 128009 # for llama3
        ):
        self.target_lang_token_ids = target_lang_token_ids
        self.num_token_ids = num_token_ids
        self.sp_token_ids = sp_token_ids
        self.thresh_diff = thresh_diff
        self.eos_id = eos_id
        self.boost_value = boost_value

        self.boost_mask = None
        self.other_mask = None

        self.target_lang_token_ids.append(eos_id)
    
    def _set_masks(self, shape):
        # set mask for the target lang token ids
        self.boost_mask = torch.zeros(shape, dtype=torch.bool, device="cuda")
        self.boost_mask[..., self.target_lang_token_ids] = True
        self.boost_mask[..., self.num_token_ids] = True
        self.boost_mask[..., self.sp_token_ids] = True

        self.other_mask = ~self.boost_mask
    
    def is_prob_flat(self, probs: torch.FloatTensor) -> bool:
        """
        Determines if the probability distribution is flat by considering the difference of 
        max probability from the two categories

        Args:
            probs (torch.FloatTensor): Prediction probabilities
        
        Retruns:
            bool: True if the distribution is flat, False otherwise
        """
        max_boost_prob, _ = (probs*self.boost_mask).max(dim=-1, keepdim=True)
        max_other_prob, _ = (probs*self.other_mask).max(dim=-1, keepdim=True)

        # calculate difference
        prob_diff = torch.abs(max_boost_prob - max_other_prob).squeeze()

        return (prob_diff <= self.thresh_diff).unsqueeze(-1)

    def __call__(
            self, 
            input_ids: torch.LongTensor, 
            scores: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        Runs when the class is called

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length)): 
                Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor of shape (batch_size, config.vocab_size)): 
                Prediction scores of a language modeling head. 
                These can be logits for each vocabulary when not using beam search 
                or log softmax for each vocabulary token when using beam search
        
        Returns:
            torch.FloatTensor of shape (batch_size, config.vocab_size)
        """
        if self.boost_mask == None:
            self._set_masks(scores.size())
        probs = torch.softmax(scores, dim=-1)
        mask = self.is_prob_flat(probs)
        boost = mask * self.boost_value
        scores[..., self.target_lang_token_ids] += boost

        return scores
