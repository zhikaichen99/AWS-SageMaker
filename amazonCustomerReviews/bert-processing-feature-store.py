import tensorflow as tf
import collections
import  json
import os
import pandas as pd
import csv
from transformers import DistilBertTokenizer

# set up distilbert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

REVIEW_BODY_COLUMN = 'review_body'
REVIEW_ID_COLUMN = 'review_id'

LABEL_COLUMN = 'star_rating'
LABEL_VALUES = [1,2,3,4,5]

label_map = {}