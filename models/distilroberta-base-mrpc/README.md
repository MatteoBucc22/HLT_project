---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3668
- loss:ContrastiveLoss
base_model: distilbert/distilroberta-base
widget:
- source_sentence: Mike Austreng , editor of the weekly Cold Spring Record , said
    he saw one wounded student taken from the school by helicopter .
  sentences:
  - Mike Austreng , the editor of the weekly Cold Spring Record , said he saw one
    wounded person airlifted from the school and another taken away by ambulance .
  - President Bush signed a waiver exempting 22 countries because they had signed
    but not yet ratified immunity agreements .
  - United said it the new airline use 40 jets , each seating 156 passengers .
- source_sentence: Most times , wholesalers sold the questionable drugs to one of
    three huge national distributors that supply virtually all drugs in the country
    .
  sentences:
  - Chante Jawaon Mallard , 27 , is charged with murder and tampering with evidence
    .
  - A female colleague , 25 , was slashed across the face as the man continued to
    try to reach the front of the plane .
  - At one point in the chain , a wholesaler would sell the questionable drugs to
    one of three huge national companies that supply virtually all drugs in this country
    .
- source_sentence: On Monday , U.S. Army Pfc. Jessica Lynch was awarded the Bronze
    Star , Purple Heart and Prisoner of War medals at Walter Reed Army Medical Center
    .
  sentences:
  - The analyst , already let go by Merrill , can request a hearing before a NASD
    panel .
  - '" Obviously , I have made statements that are ludicrous and crazy and outrageous
    because that ''s the way I was . "'
  - Lynch , who returns to the hills of West Virigina Tuesday , also received the
    Purple Heart and Prisoner of War medals at Walter Reed Army Medical Center in
    Washington .
- source_sentence: Their worst-case scenario is the accidental deletion or malicious
    falsification of ballots from the 1.42 million Californians who could vote on
    electronic touch-screen machines .
  sentences:
  - The unusual decision to declassify a major intelligence report was a bid by the
    White House to quiet a growing controversy over Bush 's allegations about Iraq
    's weapons programs .
  - Tropical Storm Claudette headed for the Texas coast Saturday , with forecasters
    saying the sluggish storm system could reach hurricane strength before landfall
    early Tuesday .
  - Their worst-case scenario is the accidental deletion or malicious falsification
    of ballots from the 1.42 million Californians voting electronically - 9.3 percent
    of the state 's 15.3 million registered voters .
- source_sentence: The 2000 Democratic platform supported " the full inclusion of
    gay and lesbian families in the life of the nation . "
  sentences:
  - In 1995 , Schwarzenegger expanded the program nationwide to 15 cities , serving
    200,000 children , most of whom come from housing projects or homeless shelters
    .
  - The Democrat 's 2000 platform didn 't explicitly support gay marriages but backed
    " the full inclusion of gay and lesbian families into the life of the nation .
    "
  - The program , which only briefly downloads the pornographic material to the usurped
    computer , is invisible to the computer 's owner .
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
- cosine_accuracy_threshold
- cosine_f1
- cosine_f1_threshold
- cosine_precision
- cosine_recall
- cosine_ap
- cosine_mcc
model-index:
- name: SentenceTransformer based on distilbert/distilroberta-base
  results:
  - task:
      type: binary-classification
      name: Binary Classification
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy
      value: 0.7941176470588235
      name: Cosine Accuracy
    - type: cosine_accuracy_threshold
      value: 0.8946959972381592
      name: Cosine Accuracy Threshold
    - type: cosine_f1
      value: 0.8604651162790697
      name: Cosine F1
    - type: cosine_f1_threshold
      value: 0.8919186592102051
      name: Cosine F1 Threshold
    - type: cosine_precision
      value: 0.8018575851393189
      name: Cosine Precision
    - type: cosine_recall
      value: 0.9283154121863799
      name: Cosine Recall
    - type: cosine_ap
      value: 0.9196327255422294
      name: Cosine Ap
    - type: cosine_mcc
      value: 0.49483719296434514
      name: Cosine Mcc
---

# SentenceTransformer based on distilbert/distilroberta-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [distilbert/distilroberta-base](https://huggingface.co/distilbert/distilroberta-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [distilbert/distilroberta-base](https://huggingface.co/distilbert/distilroberta-base) <!-- at revision fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The 2000 Democratic platform supported " the full inclusion of gay and lesbian families in the life of the nation . "',
    'The Democrat \'s 2000 platform didn \'t explicitly support gay marriages but backed " the full inclusion of gay and lesbian families into the life of the nation . "',
    'In 1995 , Schwarzenegger expanded the program nationwide to 15 cities , serving 200,000 children , most of whom come from housing projects or homeless shelters .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Binary Classification

* Evaluated with [<code>BinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.BinaryClassificationEvaluator)

| Metric                    | Value      |
|:--------------------------|:-----------|
| cosine_accuracy           | 0.7941     |
| cosine_accuracy_threshold | 0.8947     |
| cosine_f1                 | 0.8605     |
| cosine_f1_threshold       | 0.8919     |
| cosine_precision          | 0.8019     |
| cosine_recall             | 0.9283     |
| **cosine_ap**             | **0.9196** |
| cosine_mcc                | 0.4948     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,668 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | float                                                          |
  | details | <ul><li>min: 11 tokens</li><li>mean: 27.04 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 26.9 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.67</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                   | sentence_1                                                                                                                                                                              | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Associated Press Writer Sara Kugler contributed to this report from New York City .</code>                                                             | <code>Associated Press Writer Michael Gormley contributed to this report from Albany .</code>                                                                                           | <code>0.0</code> |
  | <code>With the test , Hubbard said , " I believe that we have found the smoking gun .</code>                                                                 | <code>" We have found the smoking gun , " investigating board member Scott Hubbard said .</code>                                                                                        | <code>0.0</code> |
  | <code>More than 100 officers launched the raids in the final phase of a two-year operation investigating a cocaine import and money laundering ring .</code> | <code>More than 100 police officers were involved in the busts that were the culmination of a two-year operation investigating the cocaine importing and money laundering gang .</code> | <code>1.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | cosine_ap |
|:------:|:----:|:-------------:|:---------:|
| 0.5453 | 500  | 0.0303        | 0.8660    |
| 1.0    | 917  | -             | 0.8827    |
| 1.0905 | 1000 | 0.0259        | 0.8828    |
| 1.6358 | 1500 | 0.0248        | 0.8971    |
| 2.0    | 1834 | -             | 0.9067    |
| 2.1810 | 2000 | 0.0231        | 0.9069    |
| 2.7263 | 2500 | 0.0234        | 0.9160    |
| 3.0    | 2751 | -             | 0.9170    |
| 3.2715 | 3000 | 0.0226        | 0.9178    |
| 3.8168 | 3500 | 0.0223        | 0.9202    |
| 4.0    | 3668 | -             | 0.9199    |
| 4.3621 | 4000 | 0.0219        | 0.9188    |
| 4.9073 | 4500 | 0.0218        | 0.9195    |
| 5.0    | 4585 | -             | 0.9196    |


### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.7.0+cu128
- Accelerate: 1.6.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->