import random, json, os, torch, argparse
from typing import Tuple, List, Dict
from Files.generate import generate_training_dataset
from Files.fineTune import fine_tune
from Files.evaluate import evaluate
from dataset.Dataset import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset.example_datasets.Creak import Creak
from dataset.example_datasets.Ecqa import Ecqa
from dataset.example_datasets.Bbh import Bbh
from transformers.optimization import Adafactor
from transformers import TrainingArguments, Trainer
from Files.rsi_utils.rsi_utils import str_to_bool

def select_eval_iterations(num_evals: int, total_iter: int):
    """
    Given the total number of RSI iterations and the number of evaluations, 
    return a list of indicies (iterations) at which we evaluate the model.
    """
    assert total_iter >= num_evals
    r = total_iter//num_evals
    mod = total_iter % num_evals
    counts = [1+r]*mod + [r]*(num_evals-mod)
    random.shuffle(counts)
    counts = [sum(counts[:i])-1 for i in range(1, len(counts))]
    counts.append(total_iter-1)
    return counts

def update_rsi_states(iteration, current_state):
    if not os.path.exists("rsi-states.json"):
        with open("rsi-states.json", "w") as f:
            json.dump({}, f)
    with open("rsi-states.json", "r") as f:
        states = json.load(f)
    states["iteration"] = iteration
    states["current_state"] = current_state
    with open("rsi-states.json", "w") as f:
        json.dump(states, f)

def resume_rsi_states(checkpoint_state, checkpoint_iter, curr_iter):
    """
    Determine whether we will skip, resume, or start fresh at generate, fine-tune, and eval
    """
    # if we have completed and saved the current iteration, skip all 
    if checkpoint_iter > curr_iter: 
        return "skip", "skip", "skip"
    # if we have not started the current iteration
    if checkpoint_iter < curr_iter: 
        return False, False, False
    # if checkpoint is at the current iteration
    if checkpoint_state == "generate":
        # resume generate, not-resume fine-tune, not-resume eval
        return True, False, False
    elif checkpoint_state == "fine_tune":
        # skip generate, resume fine-tune, not-resume eval
        return "skip", True, False
    elif checkpoint_state == "eval":
        # skip generate, skip fine-tune, resume evals
        return "skip", "skip", True

def rsi(N, iterations, num_evals, model, tokenizer, train_datasets: List[Tuple[Dataset, List]], eval_datasets, generate_args: Dict, train_args: Dict, eval_args: Dict, resume_from_checkpoint=False):
    """
    datasets_dics: a dictionary. Key: data objects. Value: dataset of the corresponding data object. 
    generate_args: Dict
        - batch_size: default to 16
        - num_pathways: defualt to 32
        - method: default to "cot"
        - checkpoint_dir: default to None
    train_args: Dict
        - training_args: A Huggingface TrainingArguments object.
        - optimizer: default adafactor
        - lr_scheduler: default None
        - model_output_dir: Optional[str] = None
        - resume_trainer_states: default True
    eval_args: Dict
        - batch_size
        - save_every: Optional[int] = 50
        - checkpoint_dir: Optional[str] = None
    """
    # checkpoint
    if resume_from_checkpoint:
        assert os.path.exists("rsi-states.json"), "rsi_states.json is required to resume from checkpoint but not found."
        with open("rsi-states.json", "r") as f:
            rsi_states = json.load(f)
    else:
        rsi_states = {"iteration": -1, "current_state": None}
    checkpoint_state = rsi_states["current_state"]
    checkpoint_iteration = rsi_states["iteration"]

    # initialize or load performance
    if resume_from_checkpoint:
        with open('performance.json', "r") as f:
            performance = json.load(f)
    else:  
        performance = []
        with open('performance.json', "w") as f:
            json.dump(performance, f)

    if iterations >= num_evals:
        eval_iters = select_eval_iterations(num_evals, iterations)
        for iter in range(iterations):
            print(checkpoint_state, checkpoint_iteration, iter)
            resume_generate, resume_finetune, resume_eval = resume_rsi_states(checkpoint_state, checkpoint_iteration, iter)
            if resume_generate != "skip":
                update_rsi_states(iter, "generate")
                mixture = generate_training_dataset(iter, N, model, tokenizer, train_datasets, resume_from_checkpoint=resume_generate, **generate_args)
            if resume_finetune != "skip":
                update_rsi_states(iter, "fine-tune")
                folder_path = generate_args["checkpoint_dir"] if "checkpoint_dir" in generate_args else "generate_checkpoints"
                fine_tune(f'{folder_path}/{iter}/all_data.json', model, tokenizer, resume_from_checkpoint=resume_finetune, **train_args)
            if resume_eval != "skip":
                update_rsi_states(iter, "eval")
                if iter in eval_iters:
                    metrics = evaluate(iter, eval_datasets, model, tokenizer, resume_from_checkpoint=resume_eval, **eval_args)
                    performance.append((iter*N*len(train_datasets), metrics))  
            # save performance
            with open('performance.json', "w") as f:
                json.dump(performance, f)

    return performance



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str_to_bool, default=None)
    args = parser.parse_args()
    resume = args.resume if args.resume is not None else False

    N = 30
    iterations = 2
    num_evals = 1
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto") #, cache_dir="drive/MyDrive/FLAN-T5-XXL"
    creak = Creak()
    ecqa = Ecqa()
    train_datasets = [(creak, creak.train), (ecqa, ecqa.train)]
    eval_datasets = [(Bbh(), "direct")]
    generate_args = {
        "batch_size": 8,
        "num_pathways": 32,
        "method": "cot"
    }
    train_args = {
        "optimizer": Adafactor(model.parameters(), relative_step=False, warmup_init=False, scale_parameter=False, lr=1e-3),
        "lr_scheduler": None,
        "training_args": TrainingArguments(         
            output_dir="./fine_tune_checkpoints", 
            logging_steps=50,
            num_train_epochs=3,         
            per_device_train_batch_size=16,
            save_strategy="epoch",
            save_total_limit=1,
            ),
        "resume_trainer_states": True,
    }
    eval_args = {
        "batch_size": 32,
        "save_every": 100,
        "checkpoint_dir": None
    }

    rsi(N, iterations, num_evals, model, tokenizer, train_datasets, eval_datasets, generate_args, train_args, eval_args, resume_from_checkpoint=resume)
