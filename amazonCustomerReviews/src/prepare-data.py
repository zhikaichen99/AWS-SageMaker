import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import RobertaTokenizer  # Import the tokenizer class

PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)


# Helper functions:
# Convert star rating into sentiment. Function will be used in processing data
def convert_to_sentiment(rating):
    if rating in {1,2}:
        return -1
    if rating == 3:
        return 0
    if rating in {4,5}:
        return 1

def convert_sentiment_labelid(sentiment):
    if sentiment == -1:
        return 0
    if sentiment == 0:
        return 1
    if sentiment == 1:
        return 2

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

def process_data(file, balance_dataset, max_seq_length, prefix, feature_group_name):
    df = pd.read_csv(file, index_col = 0)

    # remove null values in dataset
    df = df.dropna()
    df = df.reset_index(drop = True)

    # Convert the star rating in the dataset into sentiments
    df['sentiment'] = df['star_rating'].apply(lambda rating: convert_to_sentiment(rating))

    # Convert text/review into bert embedding
    df['input_ids'] = df['Review Text'].apply(lambda review: convert_to_bert_format(review, max_seq_length))

    # Convert the sentiments into label ids 
    df['label_ids'] = df['sentiment'].apply(lambda sentiment: convert_sentiment_labelid(sentiment))

    # Convert index into review_id
    df.reset_index(inplace =  True)
    df = df.rename(columns = {'index': 'review_id',
                             'Review Text': 'review_body'})
    
    # Keep necessary columns
    df = df[['review_id', 'sentiment', 'label_ids', 'input_ids', 'review_body']]
    df = df.reset_index(drop = True)

    # balance the dataset 
    if balance_dataset:
        # group the unbalanced dataset by sentiment class
        df_unbalanced_grouped_by = df.groupby('sentiment')
        df_balanced = df_unbalanced_grouped_by.apply(lambda x: x.sample(df_unbalanced_grouped_by.size().min()).reset_index(drop = True))

        df = df_balanced
    
    # Split data into train, test, and validation set

    # Split into training and holdout set
    df_train, df_holdout = train_test_split(df, test_size = 0.1, stratify = df['sentiment'])

    # Split the holdout set into test and validation set
    df_validation, df_test = train_test_split(df_holdout, test_size = 0.5, stratify = df['sentiment'])

    df_train = df_train.reset_index(drop = True)
    df_validation = df_validation.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    # write data to tsv file
    



if __name__ == "__main__":
    args = parse_args()
    
    

    