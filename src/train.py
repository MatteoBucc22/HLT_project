
from torch.utils.data import DataLoader

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss, SoftmaxLoss, CoSENTLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator, TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim

from tqdm.auto import tqdm
import time
import datetime
from random import sample


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from datasets import load_dataset


def train(model_name, dataset_name, element_name, peft_config=None, lr=2e-5, batch_size=16, warmup_ratio=0.1):
  # 1. Load a model to finetune with 2. (Optional) model card data
  model = SentenceTransformer(
      model_name,
      """model_card_data=SentenceTransformerModelCardData(
          language="en",
          license="apache-2.0",
          model_name="MPNet base trained on AllNLI triplets",
      )"""
  )


  is_peft = False;
  if (peft_config != None):
    model.add_adapter(peft_config)
    is_peft = True

  print(f"Launching with peft = {is_peft} lr = {lr} batch size = {batch_size} warmup_ratio = {warmup_ratio}")

  # 3. Load a dataset to finetune on
  #dataset = load_dataset("sentence-transformers/all-nli", "triplet")
  dataset = load_dataset("glue", dataset_name)
  if (len(dataset['train']) > 5000):
    train_dataset = dataset["train"].select(range(5000))
  else:
    train_dataset = dataset["train"]

  train_dataset = train_dataset.remove_columns('idx') #disturba e basta

  if (len(dataset['validation']) > 2500):
    eval_dataset = dataset["validation"].select(range(2500))
  else:
    eval_dataset = dataset["validation"]
  eval_dataset = eval_dataset.remove_columns('idx') #disturba e basta
  test_dataset = dataset["test"]

  # 4. Define a loss function
  #loss = MultipleNegativesRankingLoss(model)
  loss = ContrastiveLoss(model=model)

  # 5. (Optional) Specify training arguments
  args = SentenceTransformerTrainingArguments(
      # Required parameter:
      output_dir=f"models/{model_name}/{dataset_name}/peft-{is_peft}/{lr}/{batch_size}/{warmup_ratio}",
      # Optional training parameters:
      num_train_epochs=5,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      learning_rate=lr,
      warmup_ratio=warmup_ratio,
      fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
      bf16=False,  # Set to True if you have a GPU that supports BF16
      batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
      # Optional tracking/debugging parameters:
      eval_strategy="steps",
      eval_steps=100,
      save_strategy="steps",
      save_steps=100,
      save_total_limit=2,
      logging_steps=100,
      run_name=f"models-{model_name}-{dataset_name}-peft-{is_peft}-{lr}-{batch_size}-{warmup_ratio}",  # Will be used in W&B if `wandb` is installed
  )

  # 6. (Optional) Create an evaluator & evaluate the base model
  """dev_evaluator = TripletEvaluator(
      anchors=eval_dataset["anchor"],
      positives=eval_dataset["positive"],
      negatives=eval_dataset["negative"],
      name="all-nli-dev",
  )"""
  dev_evaluator = BinaryClassificationEvaluator(
      sentences1=eval_dataset[element_name+'1'],
      sentences2=eval_dataset[element_name+'2'],
      labels=[float(ele) for ele in eval_dataset['label']])
  dev_evaluator(model)

  # 7. Create a trainer & train
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
    #grid search
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
