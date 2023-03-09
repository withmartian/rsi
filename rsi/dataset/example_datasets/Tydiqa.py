from datasets import load_dataset
import random, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataset import Dataset

class Tydiqa(Dataset):
  name = "tydiqa"

  def __init__(self, subset = "secondary_task", random_seed = 0):
    """
    Initializes the dataset and check required attributes

    subset: "secondary_task" (GoldP) or "primary_task"
    """
    self.check_required_attributes()
    dataset = load_dataset("tydiqa", subset)
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.valid = random.sample([exp for exp in dataset["validation"]], dataset.num_rows["validation"])


  def create_prompt(self, exp, class_name: str = None, method: str = "direct"):
    """
    exp: a singular example
    method: for Tydiqa, we only use the "direct" method because baseline accuracy is high. 
    """
    question = "Context: " + exp['context']
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"


  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return exp["answers"]['text']

  
  def extract_answer(self, output):
    """
    output: a singular output
    """
    return output