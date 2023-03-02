from abc import ABC, abstractmethod, ABC
from typing import Optional, List, Dict, Tuple

class Dataset(ABC):
  name = None
  instruction = None
  cot_prompts = None
  direct_prompts = None

  @abstractmethod
  def __init__(self):
    # load the dataset and any util variables
    pass

  @abstractmethod
  def create_prompt(self, exp, method: str):
    # create the CoT or direct prompt used to generate inferences
    # exp: a singular example
    # method: "cot" or "direct"
    pass

  @abstractmethod
  def correct_answer(self, exp):
    # exp: a singular example
    pass

  @abstractmethod
  def extract_answer(self, output):
    # output: list or singular example??
    pass

  @staticmethod
  def get_batches(examples, batch_size):
    """
    Returns a generator that yiels examples in batches of batch_size

    examples: a list of dataset examples
    batch_size: batch size
    """
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]

  @staticmethod
  def generate_batched(model, tokenizer, batch, num_pathways: int, device="gpu", **gen_kwargs):
    """
    model, tokenizer: huggingface model, tokenizer
    batch: one singular batch of example
    num_pathways: number of inferences to generate per example
    device: "gpu" or "cpu"
    gen_kwargs: parameters to be passed into model.generate(); for example:
    - max_length
    - temperature
    - top_p
    """
    # Set kwargs for generation
    if "max_length" not in gen_kwargs: gen_kwargs["max_length"] = 100
    if "temperature" not in gen_kwargs:  gen_kwargs["temperature"] = 0.7
    if "top_p" not in gen_kwargs:  gen_kwargs["top_p"] = 0.95
    if num_pathways > 1: gen_kwargs["do_sample"] = True
    gen_kwargs["num_return_sequences"] = num_pathways

    if device == "gpu":
      batch_out = model.generate(batch.input_ids.to('cuda'), **gen_kwargs)
    else:
      batch_out = model.generate(batch.input_ids, **gen_kwargs)
    return [tokenizer.decode(seqs, skip_special_tokens=True) for seqs in batch_out] 


  def get_pathways(self, model, tokenizer, dataset, batch_size, num_pathways: int, num_samples: Tuple = (None, None), method: str = "direct", **gen_kwargs):
    """
    Return inference for the data passed in. shape: num_examples x num_pathways.

    model: huggingface model
    dataset: dataset (List) after randomization done in the __init__ method.
    batch_size: batch size for pathway generation
    num_pathways: number of inferences generated per example
    num_samples: (start, end) indicates the starting and ending index of the samples that we'll use to generate pathways   FIXME
    method: "direct" or "cot"
    gen_kwargs: parameters to be passed into model.generate(); for example:
      - max_length
      - temperature
      - top_p
    """

    print("-"*100)
    dataset = dataset[num_samples[0] : num_samples[1]]
    print(f'generating {len(dataset)} {self.name} samples...')
    prompts = [self.create_prompt(exp, method) for exp in dataset]
    batches = [tokenizer(batch, return_tensors="pt", padding=True) for batch in self.get_batches(prompts, batch_size)]
    result = []
    for i, batch in enumerate(batches):
      result.extend(list(self.get_batches(self.generate_batched(model, tokenizer, batch, num_pathways, **gen_kwargs), num_pathways)))
    self.last_sampled = num_samples[1]  # FIXME: do we need last_sampled?
    return result