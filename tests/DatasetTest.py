from typing import Optional, List, Dict, Tuple

class DatasetTest:
  name = "datasetTest"
  def __init__(self, Dataset, model, tokenizer):
    self.dataset = Dataset
    self.model = model
    self.tokenizer = tokenizer
    self.test_counter = {"pass": 0, "total": 0}

  def dataset_create_prompt_test(self, method: Optional[str] = "direct"):
    print("### dataset_create_prompt_test ###")
    self.test_counter["total"] += 1
    try:
      print(self.dataset.create_prompt(self.dataset.train[0], method))
      self.test_counter["pass"] += 1
    except Exception as e: 
      print(e)

  def dataset_correct_answer_test(self):
    print("### dataset_correct_answer_test ###")
    self.test_counter["total"] += 1
    try:
      print(self.dataset.correct_answer(self.dataset.train[0]))
      self.test_counter["pass"] += 1
    except Exception as e: 
      print(e)

  def dataset_extract_answer_test(self):
    print("### dataset_extract_answer_test ###")
    self.test_counter["total"] += 1
    try:
      paths = self.dataset.get_pathways(self.model, self.tokenizer, self.dataset.train, 1, 5, (0, 1), "cot")[0]
      print(paths)
      print([self.dataset.extract_answer(p) for p in paths])
      self.test_counter["pass"] += 1
    except Exception as e: 
      print(e)

  def test_all(self):
    self.dataset_create_prompt_test()
    self.dataset_correct_answer_test()
    self. dataset_extract_answer_test()
    print("\n"*3 + "#"*50)
    print(f'passed: {self.test_counter["pass"]}/{self.test_counter["total"]} tests')
