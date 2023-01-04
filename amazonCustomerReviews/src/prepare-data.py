import pandas as pd
from transformers import RobertaTokenizer  # Import the tokenizer class

PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)


# Function to convert text to required formatting
def convert_to_bert_format(text, max_seq_length):
    """
    We need to perform the following steps to our text data:
    1. Add special tokens to the start and end of each sentence.
    2. Pad & Truncate all sentences to a single constant length
    3. Differentiate real tokens from padding tokens with the 'attention mask'.
    """
    encode_plus = tokenizer.encode_plus(
        text,                         # Text to encode 
        add_special_tokens = True,    # Add '[CLS]' and '[SEP]' to start and end of sentence
        max_length = max_seq_length,  # Pad and truncate all sentences
        return_token_type_ids = False,
        pad_to_max_length = True,
        return_attention_mask = True, # Construct attention mask
        return_tensors = 'pt',        # Return PyTorch Tensor
        truncation = True
    )
    return encode_plus['input_ids'].flatten().tolist()