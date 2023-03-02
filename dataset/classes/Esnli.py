from TrainDataset import TrainDataset
from datasets import load_dataset
import random
from example import generate_5way_finetune_mixture

class Esnli(TrainDataset):
  name = "esnli"
  instruction = "Answer the following multiple choice question with options yes, no, it is not possible to tell."
  cot_prompts = """Premise: "A person on a horse jumps over a broken down airplane." 
Based on this premise, can we conclude the hypothesis "A person is at a diner, ordering an omelette." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: One jumping horse cannot be in a diner ordering food. The answer is no. 

Premise: "A person on a horse jumps over a broken down airplane." 
Based on this premise, can we conclude the hypothesis "A person is outdoors, on a horse." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: A broken down airplane is outdoors. The answer is yes. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "They are smiling at their parents." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind it. The answer is it is not possible to tell. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "The kids are frowning." is true? 
OPTIONS:
- yes 
- no 
- it is not possible to tell 
A: One cannot be smiling and frowning at the same time. The answer is no. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "There are children present." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A:The children must be present to see them smiling and waving. The answer is yes."""
  direct_prompts = """Premise: "A person on a horse jumps over a broken down airplane." 
Based on this premise, can we conclude the hypothesis "A person is training his horse for a competition." is true? OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: The answer is it is not possible to tell. 

Premise: "A person on a horse jumps over a broken down airplane." 
Based on this premise, can we conclude the hypothesis "A person is at a diner, ordering an omelette." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: The answer is no. 

Premise: "A person on a horse jumps over a broken down airplane." 
Based on this premise, can we conclude the hypothesis "A person is outdoors, on a horse." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: The answer is yes. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "They are smiling at their parents." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: The answer is it is not possible to tell. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "The kids are frowning." is true? 
OPTIONS:
- yes 
- no 
- it is not possible to tell 
A: The answer is no. 

Premise: "Children smiling and waving at camera." 
Based on this premise, can we conclude the hypothesis "There are children present." is true? 
OPTIONS: 
- yes 
- no 
- it is not possible to tell 
A: The answer is yes."""
  
  def __init__(self, generate_finetune_mixture = generate_5way_finetune_mixture, random_seed = 0):
    """
    Initializes attributes of the dataset.
    generate_finetune_mixture: a function that takes in a list of filtered inferences and return a list of fine-tune entries
    """
    self.check_required_attributes()
    super().__init__(generate_finetune_mixture)
    dataset = load_dataset("esnli")
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.test = random.sample([exp for exp in dataset["test"]], dataset.num_rows["test"])
    self.last_sampled = 0

  def create_prompt(self, exp, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    question = f'Premise: "{exp["premise"]}"'
    question += f'\nBased on this premise, can we conclude the hypothesis "{exp["hypothesis"]}" is true?'
    question += "\nOPTIONS:" + "\n- yes" + "\n- no" + "\n- it is not possible to tell"
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return exp["label"]

  def extract_answer(self, output):
    """
    output: a singular output
    """
    if output:
      pred = output.strip().split(". ")[-1]
      last_word = pred.split(" ")[-1].split(".")[0] # last word of last sentence without period
      if last_word == "yes": return 0
      if last_word == "no": return 2
      end = pred.split(" ")[-4:]
      end[-1] = end[-1].split(".")[0]    
      if end == ["not", "possible", "to", "tell"]: return 1
      return "invalid"
    return ''
