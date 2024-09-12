# rnn

[![codecov](https://codecov.io/gh/yadickson/rnn/graph/badge.svg?token=MXA5STVN07)](https://codecov.io/gh/yadickson/rnn)

Red neuronal basica con python

## Install python 3.10 on Mac

```bash
brew install python@3.10
```

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


[referencia](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/)

[referencia](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)
