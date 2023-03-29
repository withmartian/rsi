from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
import re, random, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataset import Dataset

class Strategyqa(Dataset):
  name = "strategyqa"
  instruction = "Answer the following yes/no question."
  cot_prompts = """Q: Do hamsters provide food for any animals? 
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes. 

Q: Could Brooke Shields succeed at University of Pennsylvania? 
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is yes. 

Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? 
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. The answer is no. 

Q: Yes or no: Is it common to see frost during some college commencements? 
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. The answer is yes. 

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? 
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. The answer is no. 

Q: Yes or no: Would a pear sink in water? 
A: The density of a pear is about 0.6 g/cm3 , which is less than water. Objects less dense than water float. Thus, a pear would float. The answer is no."""
  direct_prompts = """Q: Do hamsters provide food for any animals? 
A: The answer is yes. 

Q: Could Brooke Shields succeed at University of Pennsylvania? 
A: The answer is yes. 

Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls? 
A: The answer is no. 

Q: Yes or no: Is it common to see frost during some college commencements? 
A: The answer is yes. 

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)? 
A: The answer is no. 

Q: Yes or no: Would a pear sink in water? 
A: The answer is no."""

  def __init__(self, random_seed = 0):
    """
    Initializes attributes of the dataset
    """
    self.check_required_attributes()
    random.seed(random_seed)
    train = load_dataset("metaeval/strategy-qa")
    test = load_dataset("amydeng2000/strategy-qa")
    self.train = random.sample([exp for exp in train["train"]], train.num_rows["train"])
    self.test = random.sample([exp for exp in test["test"]], test.num_rows["test"])

  def get_question(self, exp):
    return exp["question"]

  def create_prompt(self, exp, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    question = self.get_question(exp)
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return bool(exp['answer'])

  def extract_answer(self, output):
    """
    This function converts a yes/no answer to a true/false answer to match the 
    ground truth answer provided in the dataset

    output: one singular output
    """
    if output:
      pred = output.strip().split(". ")[-1]
      pred = re.split('\s|\.', pred)  # split by space and period
      if "yes" in pred and "no" in pred: # if both 'yes' and 'no' are present, take the first one
        if pred.index("yes") < pred.index("no"): return True
        else: return False
      elif "yes" in pred: 
        return True
      elif "no" in pred:
        return False
      else:
        return "invalid"
    return ''