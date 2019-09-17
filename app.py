import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
from pymongo import MongoClient

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer, XLMConfig,
                                  XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification,
                                  XLNetTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import InputExample, convert_examples_to_features
from scipy.special import softmax
import json


input_dir = 'manual/'
classifier = 'outputs/'
classifier_vocab = 'outputs/'
max_seq_len = 512
output_mode = 'classification'
eval_batch_size = 1
model_type = 'roberta'

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)

MODEL_CLASSES = {
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

tokenizer = tokenizer_class.from_pretrained(classifier)
model = model_class.from_pretrained(classifier)
model.eval()

client = MongoClient()
db = client['pseudo-results']
collection = db['results']

def tokenize(sentence):
    sentence = sentence.replace('\n', '')
    test_examples = [InputExample(0, sentence, None, '0')]
    label_list = ["0", "1"]

    num_labels = len(label_list)
    test_examples_len = len(test_examples)
    label_map = {label: i for i, label in enumerate(label_list)}

    test_features = convert_examples_to_features(test_examples, label_list,
            max_seq_len, tokenizer, output_mode,
            cls_token_at_end=bool('model_type' == 'xlnet'),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if 'model_type' == 'xlnet' else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool('model_type' == 'roberta'),
            pad_on_left=True,                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id= 4 if 'model_type' == 'xlnet' else 0)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    test_sampler = SequentialSampler(test_data)
    eval_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

    for batch in eval_dataloader:
        batch = tuple(t for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3]}
    return inputs

class Prediction(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument("sentence", type=str, required=True)
        self.reqparse.add_argument("choice", type=str, required=True)
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        text = args['sentence']
        
        preds, probs = get_prediction(tokenize(text))

        if preds:
            is_pseudo = "Pseudoscience detected"
        else:
            is_pseudo =  "Pseudoscience not detected"

        user_prediction = args['choice']
        if user_prediction == "true":
            user_prediction = True
        elif user_prediction == "false":
            user_prediction = False
        else:
            user_prediction = None

        if user_prediction is not None:
            collection.insert_one(
                {
                    "text": text,
                    "probs": float(probs),
                    "prediction": int(preds),
                    "user_prediction": int(user_prediction)
                }
            )

        return {"pseudoscience": is_pseudo, "probability": str(probs)}


def get_prediction(inputs):
    outputs = model(**inputs)
    _, logits = outputs[:2]
    preds = logits.detach().numpy()
    probs = softmax(preds, axis=1)[0, 1].item()
    preds = np.argmax(preds, axis=1)

    return preds, probs



api.add_resource(Prediction, '/api/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
