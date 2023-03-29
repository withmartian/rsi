import torch, os
from typing import Dict, List
from transformers.optimization import Adafactor
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from typing import Optional, List, Dict


def add_eos_to_example(example):
  """
  Add end of sentence token to a singular training example.
  example: A training example. A huggingface dataset entry with 'input' and 'target' as keys.
  """
  example['input'] = f'{example["input"]} </s>'
  example['target'] = f'{example["target"]} </s>'
  return example

def convert_batch_to_features(example_batch, tokenizer):
  """
  Convert a batch of training examples into their encodings
  """
  input_encodings = tokenizer.batch_encode_plus(example_batch['input'], padding='max_length', max_length=660, truncation=True) # padding = True
  target_encodings = tokenizer.batch_encode_plus(example_batch['target'], padding='max_length', max_length=122, truncation=True)
  encodings = {
      'input_ids': input_encodings['input_ids'], 
      'attention_mask': input_encodings['attention_mask'],
      'labels': target_encodings['input_ids'],
      'decoder_attention_mask': target_encodings['attention_mask']
  }
  return encodings

def preprocess(dataset, tokenizer):
  """
  Prepare a dataset for fine tuning.
  dataset: a huggingface dataset entry with 'input' and 'target' as keys.
  """
  dataset = dataset.map(add_eos_to_example)
  dataset = dataset.map(convert_batch_to_features, batched=True, fn_kwargs={"tokenizer": tokenizer})
  columns = ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']
  dataset.set_format(type='torch', columns=columns)
  # torch.save(dataset, f'train_data_epoch_{i}.pt')
  return dataset

class T2TDataCollator():
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch. Returns a dictionary of tensors.
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['labels'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }
    
# load scheduler and optimizer from checkpoint folder
def resume(trainer, training_args):
    print("--> loading previous optimizer and scheduler states...")
    checkpoint = [f for f in os.listdir("./fine_tune_checkpoints") if f and f[:10] == 'checkpoint'][0]
    output = training_args.output_dir
    trainer.optimizer.load_state_dict(torch.load(f'{output}/{checkpoint}/optimizer.pt'))
    if trainer.lr_scheduler:
      trainer.lr_scheduler.load_state_dict(torch.load(f'{output}/{checkpoint}/scheduler.pt'))
    return trainer
    
def fine_tune(dataset_file_path, model, tokenizer, resume_from_checkpoint=False, training_args=None, optimizer=None, lr_scheduler=None, resume_trainer_states=True):
    """
    Fine tunes and saves the model.
    """
    train_dataset = load_dataset("json", data_files=dataset_file_path, split="train")
    train_dataset = train_dataset.shuffle(seed=0)
    train_dataset = preprocess(train_dataset, tokenizer)

    if not optimizer: # FIXME: defualt optimizer
        optimizer = Adafactor(
                model.parameters(),
                relative_step=False,
                warmup_init=False,
                scale_parameter=False,
                lr=1e-3)
    
    if not training_args: # FIXME: default training_args
        training_args = TrainingArguments(         
                output_dir="./fine_tune_checkpoints", 
                logging_steps=50,
                num_train_epochs=3,         
                per_device_train_batch_size=16,
                save_strategy="epoch",
                save_total_limit=1,
                )
    
    trainer = Trainer(model, training_args, train_dataset=train_dataset, 
                        data_collator=T2TDataCollator(), tokenizer=tokenizer,
                        optimizers=(optimizer, lr_scheduler)
                        )
    if os.listdir("./fine_tune_checkpoints"):
      if resume_from_checkpoint:
        trainer.train(True)
        return model
      elif resume_trainer_states:
        trainer = resume(trainer, training_args)

    trainer.train()
    trainer.save_model()
    return model