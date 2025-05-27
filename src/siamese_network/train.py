
from torch.utils.data import DataLoader

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.training_args import BatchSamplers

from random import sample


from peft import LoraConfig, TaskType

from datasets import load_dataset


def train(model_name, dataset_name, element_name, peft_config=None, lr=2e-5, batch_size=16, warmup_ratio=0.1):
  model = SentenceTransformer(
      model_name
  )


  is_peft = False
  if (peft_config != None):
    model.add_adapter(peft_config)
    is_peft = True

  print(f"Launching with peft = {is_peft} lr = {lr} batch size = {batch_size} warmup_ratio = {warmup_ratio}")


  dataset = load_dataset("glue", dataset_name)
  if (len(dataset['train']) > 5000):
    train_dataset = dataset["train"].select(range(5000))
  else:
    train_dataset = dataset["train"]

  train_dataset = train_dataset.remove_columns('idx') 

  if (len(dataset['validation']) > 2500):
    eval_dataset = dataset["validation"].select(range(2500))
  else:
    eval_dataset = dataset["validation"]
  eval_dataset = eval_dataset.remove_columns('idx') 


  loss = ContrastiveLoss(model=model)

  args = SentenceTransformerTrainingArguments(
      output_dir=f"models/{model_name}/{dataset_name}/peft-{is_peft}/{lr}/{batch_size}/{warmup_ratio}",
      num_train_epochs=5,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      learning_rate=lr,
      warmup_ratio=warmup_ratio,
      fp16=True,  
      bf16=False,  
      batch_sampler=BatchSamplers.NO_DUPLICATES,  
      eval_strategy="steps",
      eval_steps=100,
      save_strategy="steps",
      save_steps=100,
      save_total_limit=2,
      logging_steps=100,
      run_name=f"models-{model_name}-{dataset_name}-peft-{is_peft}-{lr}-{batch_size}-{warmup_ratio}",  
  )

  
  dev_evaluator = BinaryClassificationEvaluator(
      sentences1=eval_dataset[element_name+'1'],
      sentences2=eval_dataset[element_name+'2'],
      labels=[float(ele) for ele in eval_dataset['label']])
  dev_evaluator(model)

  trainer = SentenceTransformerTrainer(
      model=model,
      args=args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      loss=loss,
      evaluator=dev_evaluator,
  )
  trainer.train()
  return model


if __name__ == "__main__":
    for model_name in [ 'sentence-transformers/all-MiniLM-L6-v2', 'roberta-base']: 
        for (database_name, element_name) in [('mrpc', 'sentence'), ('qqp', 'question')]: 
            for peft_config in [None, LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    target_modules=["query", "value"]
                )]:
                for lr in [1e-5, 2e-5, 3e-5]:
                    for batch_size in [16, 32, 64]:
                        for warmup_ratio in [0., 0.1, 0.2]:
                            train(model_name, database_name, element_name, peft_config, lr, batch_size, warmup_ratio)
