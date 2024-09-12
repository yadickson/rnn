# rnn

![version](https://img.shields.io/github/v/release/yadickson/rnn?style=flat-square)
![license](https://img.shields.io/github/license/yadickson/rnn?style=flat-square)
[![build](https://img.shields.io/github/actions/workflow/status/yadickson/rnn/python-app.yml?branch=main&style=flat-square)](https://github.com/yadickson/rnn/actions/workflows/python-app.yml)
![tests](https://img.shields.io/endpoint?style=flat-square&url=https%3A%2F%2Fgist.githubusercontent.com%2Fyadickson%2F2edc636fc2ff6aff4b056d455f3290be%2Fraw%2Frnn-junit-tests.json)
![coverage](https://img.shields.io/endpoint?style=flat-square&url=https%3A%2F%2Fgist.githubusercontent.com%2Fyadickson%2F2edc636fc2ff6aff4b056d455f3290be%2Fraw%2Frnn-cobertura-coverage.json)

Required python 3.10

## Make environment

```bash
python3 -m venv .venv
```

## Activate environment

```bash
source .venv/bin/activate
```

## Install poetry

```bash
pip install poetry
```

## Install dependencies

```bash
poetry install
```

## Format code

```bash
poetry exec format
```

## Check code

```bash
poetry exec lint
```

## Run test

```bash
poetry exec test
```

## Run coverage

```bash
poetry exec test:coverage
```

## Run mutation test

The mutation test fail with python 3.12

```bash
poetry exec test:mutation
```

## Training

```bash
poetry exec test:log
```

```bash
TRAINING_TEST=run poetry exec test:log tests/training/test_network_xor_training.py
```

```bash
TRAINING_TEST=run poetry exec test:log tests/training/test_network_keras_image_training.py
```

```bash
TRAINING_TEST=run poetry exec test:log tests/training/test_network_circle_shape_training.py
```

## Deactivate environment

```bash
deactivate
```


[reference](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/)

[reference](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)
