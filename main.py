from rsi.dataset.example_datasets.Aqua import Aqua
inferences = ['(A)', '(B)', '(C)', '(D)']
aqua = Aqua()
out = aqua.filter_generated_paths(aqua.train[0], inferences)
print(out)
