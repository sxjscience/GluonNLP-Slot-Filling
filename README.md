# Joint Intent Detection and Slot Filling by GluonNLP


## Introduction
Intent detection and slot filling are two essential problems in Natural Language Understanding (NLU).
In _intent detection_, the agent needs to detect the intention that the speaker's utterance conveys.
 For example, when the speaker says "Book a flight from Long Beach to Seattle", the intention is to book a flight ticket.
In _slot filling_, the agent needs to extract the semantic entities that are related to the intent. In our previous example,
"Long Beach" and "Seattle" are two semantic constituents related to the flight, i.e., the origin and the destination.

Essentially, _intent detection_ can be viewed as a sequence classification problem and _slot filling_ can be viewed as a
sequence tagging problem similar to Named-entity Recognition (NER). Due to their inner correlation, these two tasks are usually
trained jointly with a multi-task objective function.  

Here's one example of the ATIS dataset

| Sentence  | Tags | Intent Label |
| --------- | ---- | ------------ |
|    are    | O    |    atis_flight |
| there     | O    |  |
| any       | O    |  |
| flights   | O    |  |
| from      | O    |  |
| long      | B-fromloc.city_name |  |
| beach     | I-fromloc.city_name |  |
| to        | O                   |  |
| columbus  | B-toloc.city_name   |  |
| on        | O                   |  |
| wednesday | B-depart_date.day_name    |  |
| april     | B-depart_date.month_name  |  |
| sixty     | B-depart_date.day_number  |  |



In this example, we demonstrate how to use GluonNLP to build a model to perform joint intent detection and slot filling. We 
choose to finetune a pretrained BERT model.  We use two datasets [ATIS](https://github.com/yvchen/JointSLU) and [SNIPS](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines).
 
## Requirements

```
mxnet
gluonnlp
seqeval
```

You may use pip or other tools to install these packages

## Experiment
For the ATIS dataset, use the following command to run the experiment:
```bash
python demo.py --gpu 0 --dataset atis
```

It produces the final slot filling F1 = `95.98%` and intent detection accuracy = `98.54%`

For the SNIPS dataset, use the following command to run the experiment:
```bash
python demo.py --gpu 0 --dataset snips
```
It produces the final slot filling F1 = `95.76%` and intent detection accuracy = `98.71%`

Also, we train the models with three random seeds and report the mean/std

For ATIS

| Models | Intent Detection Acc (%) | Slot F1 (%) |
| ------ | ------------------------ | ----------- |
| [Intent Gating & self-attention](https://www.aclweb.org/anthology/D18-1417) | 98.77 | 96.52 |
| [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 97.42 | 95.62 |
| [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) |  97.5 | 96.1 |
| Ours | 98.69±0.11  | 95.71±0.19 |

For SNIPS

| Models | Intent Detection Acc (%) | Slot F1 (%) |
| ------ | ------------------------ | ----------- |
| [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 99.29 | 93.90 |
| [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) | 98.60 | 97.00 |
| Ours | 98.67±0.18 | 95.64±0.10 |
