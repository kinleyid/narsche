
# Measuring narrative schematicity

[![codecov](https://codecov.io/github/kinleyid/narsche/graph/badge.svg?token=EHCYVTZWCE)](https://codecov.io/github/kinleyid/narsche)

## Usage

```python
import narsche

mod = narsche.read_vectors('/path/to/vectors.txt') # Load vectors from txt file
txt = ['I sat on the sofa in my living room with a lamp'] # Example text
tokenizer = narsche.Tokenizer() # Initialize tokenizer
words = tokenizer.tokenize(txt) # Tokenize words
words = mod.keep_known(words) # Use only those words that are in the model
topic = narsche.identify_topic(words) # Identify the topic
# Compute schematicity
narsche.schematicity(
	words=words,
	model=mod,
	method='on-topic-ppn',
	topic=topic)
```
