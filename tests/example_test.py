from transformers import T5Tokenizer, T5ForConditionalGeneration
from DatasetTest import DatasetTest
import torch

import sys
sys.path.append("..")
from rsi.dataset.example_datasets.Bbh import Bbh

bbh = Bbh()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto")

test = DatasetTest(bbh, model, tokenizer)
test.test_all()