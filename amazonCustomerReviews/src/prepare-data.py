import pandas as pd
from transformers import RobertaTokenizer

PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)

def convert_to_bert_format(text_data, max_seq_length):
    """
    We need to perform the following steps to our text data:
    1. Add special tokens to the start and end of each sentence.
    2. Pad & Truncate all sentences to a single constant length
    3. Differentiate real tokens from padding tokens with the 'attention mask'.
    """