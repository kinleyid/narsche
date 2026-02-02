
# Measuring narrative schematicity

[![codecov](https://codecov.io/github/kinleyid/narsche/graph/badge.svg?token=EHCYVTZWCE)](https://codecov.io/github/kinleyid/narsche)

Methods from the paper "Computational Tools for Quantifying Schemas in Autobiographical Narratives".

## Installation

```
pip install narsche
```

`narsche` depends on `networkx` (for network models), `SpaCy` (for tokenization), and `wordfreq` for automated topic identification. Additionally, one of `SpaCy`'s models must be downloaded for `SpaCy`-based tokenization:

```
python -m spacy download en_core_web_sm
```

## Usage

### Loading and saving models

A text file of word vectors can be read using the `read_vectors()` function:

```python
vec_mod = narsche.read_vectors('/path/to/vectors.txt')
```

This produces a vector model. The text file must be formatted such that the first token (space-delimited) on a line is the word for which the remaining tokens are the vector components. This is how, for example, the [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) are formatted.

Initializing a network model requires first loading a `networkx.Graph` object:

```python
import networkx as nx

graph = nx.load('/path/to/graph')
net_mod = narsche.NetworkModel(graph)
```

A script for setting up a network model us can be found [here](/examples/create-conceptnet-graph.py).

Models can be saved using the `save()` method and loaded using the `load()` class method:

```python
net_mod.save('network.mod')
net_mod = narsche.NetworkModel.load('network.mod')

vec_mod.save('vector.mod')
vec_mod = narsche.VectorModel.load('vector.mod')
```

These are just wrappers around `pickle.[load/dump]`. Any extension can be used.

### Tokenizing narratives

Before schematicity can be computed, narratives must be tokenized, i.e., converted to a list of tokens. For this, there is a `Tokenizer()` class that relies on `SpaCy`:

```python
txt = 'I sat on the sofa in my living room with a lamp' # Example text
tokenizer = narsche.Tokenizer('en_core_web_sm') # Initialize tokenizer
words = tokenizer.tokenize(txt) # Tokenize words
words = vec_mod.keep_known(words) # Use only those words that are in the model
```

### Computing schematicity

Given a model and a set of tokens (and possibly a topic word), schematicity can be computed using the `schematicity()` function:

```python
topic = narsche.identify_topic(words) # Identify the topic
# Compute schematicity
narsche.schematicity(
	words=words,
	model=mod,
	method='on-topic-ppn', # or topic-relatedness, pairwise-relatedness, or component-size
	topic=topic)
```

See the documentation of the `schematicity()` function for kewords required by other methods.
