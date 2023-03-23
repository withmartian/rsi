import json, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_utils import extract_last_word
from Dataset import Dataset

class Mmlu(Dataset):
  name = "mmlu"
  classes = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
  with open("cot_prompts/mmlu_cot.json", "r") as f:
    cot_prompts = json.load(f)
  with open("mmlu_random_50*57_samples.json", "r") as f:
    shortened_dataset = json.load(f)

  def __init__(self, random_seed = 0):
    """
    Initializes the dataset and check required attributes.
    """
    self.check_required_attributes()
    # FIXME: sort out how to load class dataset on demand
    # self.train = {}
    # self.test = {}
    # random.seed(random_seed)
    # for c in self.classes:
    #   dataset = load_dataset("hendrycks_test", c)
    #   self.train[c] = random.sample([exp for exp in dataset["auxiliary_train"]], dataset.num_rows["auxiliary_train"])
    #   self.test[c] = random.sample([exp for exp in dataset["test"]], dataset.num_rows["test"])

  def get_question(self, exp):
    question = "Q: " + exp['question'] + "\n"
    question += f'(A) {exp["choices"][0]} (B) {exp["choices"][1]} (C) {exp["choices"][2]} (D) {exp["choices"][3]}'
    return question

  def create_prompt(self, exp, class_name: str = None, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    assert class_name != None
    question = self.get_question(exp)
    if method == "cot":
      cot_prompts = self.cot_prompts[class_name]
      return cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    keys = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
    return keys[exp['answer']]

  def extract_answer(self, output):
    """
    output: a singular output
    """
    return extract_last_word(output)