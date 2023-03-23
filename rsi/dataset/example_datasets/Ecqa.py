from datasets import load_dataset
import random, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_utils import  extract_last_word
from Dataset import Dataset

class Ecqa(Dataset):
  name = "ecqa"
  instruction = "Answer the following multiple choice question with options (a), (b), (c), (d), (e)."
  cot_prompts = """Q: If you want to set a romantic atmosphere you might light a candle where? Answer Choices: (a)dimly lit room (b)synagogue (c)bedroom (d)birthday cake (e)rosesA: A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses. The answer is (a).

Q: What might the result of unwanted flirting be? Answer Choices: (a)attraction (b)problems (c)the gallows (d)being slapped (e)curiosity
A: Person can be slapped if he does unwanted flirting to someone. Attraction cannot be the result of unwanted flirting. Unwanted flirting doesn't always create problems. The gallows or curiosity is not something that can be the result of unwanted flirting. The answer is (d).

Q: He had a lot on his plate opening business, this caused a lot of what? Answer Choices:  (a)headaches (b)making money (c)success (d)failure (e)stress
A: When someone has lot on plate, they often feel stressed. A new business demands lot o fwork that can cause stress. All the other options are incorrect as they are not a result of being a lot on plate in a business. The answer is (e)."""
  direct_prompts = """Q: What might a person see at the scene of a brutal killing? Answer Choices: (a)bloody mess (b)pleasure (c)being imprisoned (d)feeling of guilt (e)cake
A: The answer is (a).

Q: If you want to set a romantic atmosphere you might light a candle where? Answer Choices: (a)dimly lit room (b)synagogue (c)bedroom (d)birthday cake (e)roses
A: The answer is (a).

Q: What might the result of unwanted flirting be? Answer Choices: (a)attraction (b)problems (c)the gallows (d)being slapped (e)curiosity
A: The answer is (d).

Q: He had a lot on his plate opening business, this caused a lot of what? Answer Choices:  (a)headaches (b)making money (c)success (d)failure (e)stress
A: The answer is (e)."""

  def __init__(self, random_seed = 0):
    """
    Initializes attributes of the dataset.
    """
    self.check_required_attributes()
    dataset = load_dataset("yangdong/ecqa")
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.test = random.sample([exp for exp in dataset["test"]], dataset.num_rows["test"])
    self.last_sampled = 0

  def get_question(self, exp):
    question = exp["q_text"] + " "
    question += "Answer Choices:"
    question += " " + "(a)" + exp["q_op1"]
    question += " " + "(b)" + exp["q_op2"]
    question += " " + "(c)" + exp["q_op3"]
    question += " " + "(d)" + exp["q_op4"]
    question += " " + "(e)" + exp["q_op5"]
    return question

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
    options = [exp["q_op1"], exp["q_op2"], exp["q_op3"], exp["q_op4"], exp["q_op5"]]
    answer = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    return answer[options.index(exp["q_ans"])]

  def extract_answer(self, output):
    """
    output: a singular output
    """
    return extract_last_word(output)
