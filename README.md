# PyTorch Wrapper

![Logo](logo.png)

PyTorch Wrapper is a library that provides a systematic and extensible way to build, train, evaluate, and tune deep learning models
using PyTorch. It also provides several ready to use modules and functions for fast model development.

| Branch | Build | Coverage |
| :---: | :---: | :---: |
| Master | [![Build Status](https://travis-ci.com/jkoutsikakis/pytorch-wrapper.svg?branch=master)](https://travis-ci.com/jkoutsikakis/pytorch-wrapper) | [![Coverage Status](https://coveralls.io/repos/github/jkoutsikakis/pytorch-wrapper/badge.svg?branch=develop)](https://coveralls.io/github/jkoutsikakis/pytorch-wrapper?branch=master) |
| Develop | [![Build Status](https://travis-ci.com/jkoutsikakis/pytorch-wrapper.svg?branch=develop)](https://travis-ci.com/jkoutsikakis/pytorch-wrapper)| [![Coverage Status](https://coveralls.io/repos/github/jkoutsikakis/pytorch-wrapper/badge.svg?branch=develop)](https://coveralls.io/github/jkoutsikakis/pytorch-wrapper?branch=develop) |


## Installation

### From PyPI
```bash
pip install pytorch-wrapper
```

### From Source

```bash
git clone https://github.com/jkoutsikakis/pytorch-wrapper.git
cd pytorch-wrapper
pip install .
```

## Basic usage pattern

```python
import torch
import pytorch_wrapper as pw

train_dataloader = ...
val_dataloader = ...
dev_dataloader = ...

evaluators = { 'acc': pw.evaluators.AccuracyEvaluator(), ... }
loss_wrapper = pw.loss_wrappers.GenericPointWiseLossWrapper(torch.nn.BCEWithLogitsLoss())

model = ...

system = pw.System(model=model, device=torch.device('cuda'))

optimizer = torch.optim.Adam(system.model.parameters())

system.train(
    loss_wrapper,
    optimizer,
    train_data_loader=train_dataloader,
    evaluators=evaluators,
    evaluation_data_loaders={'val': val_dataloader},
    callbacks=[
        pw.training_callbacks.EarlyStoppingCriterionCallback(
            patience=3,
            evaluation_data_loader_key='val',
            evaluator_key='acc',
            tmp_best_state_filepath='current_best.weights'
        )
    ]
)

results = system.evaluate(dev_dataloader, evaluators)

predictions = system.predict(dev_dataloader)

system.save_model_state('model.weights')
system.load_model_state('model.weights')

```

## Docs & Examples

The docs can be found [here](https://pytorch-wrapper.readthedocs.io/en/latest/).

There are also the following examples in notebook format:

1. [Two Spiral Task](examples/1_two_spiral_task.ipynb)
2. [Image Classification Task](examples/2_image_classification_task.ipynb)
3. [Tuning Image Classifier](examples/3_tuning_image_classifier.ipynb)
4. [Text Classification Task](examples/4_text_classification_task.ipynb)
5. [Token Classification Task](examples/5_token_classification_task.ipynb)
6. [Text Classification Task using BERT](examples/6_text_classification_task_using_bert.ipynb)
7. [Custom Callback](examples/7_custom_callback.ipynb)
8. [Custom Loss Wrapper](examples/8_custom_loss_wrapper.ipynb)
9. [Custom Evaluator](examples/9_custom_evaluator.ipynb)
