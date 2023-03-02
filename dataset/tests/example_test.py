from transformers import T5Tokenizer, T5ForConditionalGeneration
import DatasetTest
import torch
import sentencepiece
import accelerate

import sys
sys.path.append("..")
from classes.Gsm8k import Gsm8k

gsm8k = Gsm8k()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto")

test = DatasetTest(gsm8k, model, tokenizer)
test.test_all()