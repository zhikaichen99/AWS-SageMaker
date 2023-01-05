import pandas as pd
import sagemaker
import boto3

from time import gmtime, strftime, sleep


from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer  # Import the tokenizer class


PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)


# Helper functions:

# Convert object data types to string
def object_to_string(dataframe):
    for column in dataframe.columns:
        if dataframe.dtypes[column] == 'object':
            dataframe[column] = dataframe[column].astype('str').astype('string')
    return dataframe

# Convert star rating into sentiment. Function will be used in processing data
def convert_to_sentiment(rating):
    if rating in {1,2}:
        return -1
    if rating == 3:
        return 0
    if rating in {4,5}:
        return 1

# Sentiment to label id
def convert_sentiment_labelid(sentiment):
    if sentiment == -1:
        return 0
    if sentiment == 0:
        return 1
    if sentiment == 1:
        return 2

def create_feature_group(feature_group_name, prefix):
    """
    A FeatureGroup is the main Feature Store resource that contains the metadata for all the data
    stored in Amazon SageMaker Feature Store.
    """

    
    """
    A FeatureDefinition constists of a name and one of the following data types:
    -Integral
    -String
    -Fraction
    """

    # Feature Definitions for the records
    feature_definitions = [
        FeatureDefinition(feature_name = 'review_id', feature_type = FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name = 'date', feature_type = FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name = 'sentiment', feature_type = FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name = 'label_ids', feature_type = FeatureTypeEnum.STRING), 
        FeatureDefinition(feature_name = 'input_ids', feature_type = FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name = 'review_body', feature_type = FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name = 'split_type', feature_type = FeatureTypeEnum.STRING)
    ]

    # Feature Group
    feature_group = FeatureGroup(
        name = feature_group_name,
        feature_definitions = feature_definitions,
        sagemaker_session = sagemaker_session
    )


    record_identifier_name = 'review_id'
    event_time_feature_name = 'date'

    # Create Feature Group
    feature_group.create(
        s3_uri = f's3://{bucket}/{prefix}',
        record_identifier_name = record_identifier_name,
        event_time_feature_name = event_time_feature_name,
        role_arn = role,
        enable_online_store = False
    )

    return feature_group



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


def process_data(file, balance_dataset, max_seq_length, feature_group_name):
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

    # Adding date feature into column to keep track of the data
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    df['date'] = timestamp
    
    # Split data into train, test, and validation set

    # Split into training and holdout set
    df_train, df_holdout = train_test_split(df, test_size = 0.1, stratify = df['sentiment'])

    # Split the holdout set into test and validation set
    df_validation, df_test = train_test_split(df_holdout, test_size = 0.5, stratify = df['sentiment'])

    df_train = df_train.reset_index(drop = True)
    df_validation = df_validation.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    # write data to tsv file
    output_file_path = 'processing/output/sentiment'
    df_train.to_csv('../data/{}/train/training_data_processed.tsv'.format(output_file_path))
    df_validation.to_csv('../data/{}/validation/validation_data_processed.tsv'.format(output_file_path))
    df_test.to_csv('../data/{}/test/test_data_processed.tsv'.format(output_file_path))

    column_names = ['review_id', 'sentiment', 'date', 'label_ids', 'input_ids', 'review_body']

    df_train_records = df_train[column_names]
    df_train_records['split_type'] = 'train'

    df_validation_records = df_validation[column_names]
    df_validation_records['split_type'] = 'validation'

    df_test_records = df_test[column_names]
    df_test_records['split_type'] = 'test'

    df_train_records = object_to_string(df_train_records)
    df_validation_records = object_to_string(df_validation_records)
    df_test_records = object_to_string(df_test_records)

    # Create the feature group
    feature_group = create_feature_group(feature_group_name, prefix)

    # Ingest data into SageMaker Feature Store
    """
    Ingestion is the act of populating feature groups in the feature store.
    """

    feature_group.ingest(data_frame = df_train_records, max_workers = 3, wait = True)
    feature_group.ingest(data_frame = df_validation_records, max_workers = 3, wait = True)
    feature_group.ingest(data_frame = df_test_records, max_workers = 3, wait = True)


    





if __name__ == "__main__":
    args = parse_args()
    
    

    