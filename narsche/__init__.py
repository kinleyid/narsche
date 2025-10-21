import pickle
import numpy as np
from wordfreq import word_frequency
from collections import Counter
import networkx as nx
import spacy
import sys
from pdb import set_trace

epsilon = sys.float_info.epsilon


def get_pairs(iterable):
    """
    Gets adjacent pairs of items from some iterable collection.

    Args:
        iterable (iterable): some iterable collection (e.g., a list or tuple)

    Returns:
        pairs: adjacent pairs or items in a list
    """
    pairs = [(a, b) for i, a in enumerate(iterable) for b in iterable[(i + 1) :]]
    return pairs


def identify_topic(words):
    """
    Identifies the topic using tf-idf

    Args:
        words (list): a list of words

    Returns:
        topic (str): the topic word
    """
    bag = np.array(sorted(list(set(words))))  # Unique words

    """
    # Compute tf-idf
    tf = np.array([words.count(word) for word in bag])
    df = np.array([word_frequency(word=word, lang='en') for word in bag])
    idf = np.log(1 / df)
    tf_idf = tf*idf
    """

    tf = Counter(words)
    # Compute doc frequency (could be 0)
    df = {word: word_frequency(word=word, lang="en", minimum=epsilon) for word in bag}
    # Recompute bag
    bag = np.array([word for word in df if df[word] > 0])
    idf = {word: np.log(1 / df[word]) for word in bag}
    tf_idf = np.array([tf[word] * idf[word] for word in bag])
    # Sort
    sort_idx = np.argsort(-tf_idf)  # Negative to be in descending
    sorted_bag = bag[sort_idx]
    # Get topic (top ranked)
    return str(sorted_bag[0])


def read_vectors(file, encoding="utf-8"):
    """
    Reads word vectors from a text file. Each line of the file should be formatted <word> <dim1> <dim2> <dim3> ... Vectors are automatically normalized

    Args:
        file (str): name of text file
        encoding (str): encoding used when reading the file

    Returns:
        model (VectorModel): vectors in a VectorModel
    """
    words = []
    vectors = []
    with open(file, "r", encoding=encoding) as f:
        for line in f:
            # First item in space-delimited line is token, remaining items are vector elements
            split_line = line.rstrip("\n").split(" ")
            words.append(split_line[0])
            # Normalize vector for fast dot product-based cosine similarity computation
            vector = np.asarray(split_line[1:]).astype(np.float32)
            # vector /= np.linalg.norm(vector)
            vectors.append(vector)
    vectors = np.array(vectors)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms
    return VectorModel(words, vectors)


class Model:
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected instance of %s, got %s" % (cls.__name__, type(obj).__name__)
            )
        return obj

    def keep_known(self, words):
        return [word for word in words if word in self]


class VectorModel(Model):
    """
    Vector-based model

    Attributes:
        words (list of str): words for which the model has vectors
        vectors (matrix): numpy matrix containing the vectors (in the same order as the word list)
    """

    def __init__(self, words, vectors):
        # Store efficiently---list of words, matrix of vectors, and index
        self.words = words
        self.vectors = vectors

    def __contains__(self, word):
        return word in self.words

    def in_model(self, word):
        return word in self.words

    def compute_sim(self, word1, word2):
        """
        Compute cosine similarity between words

        Args:
            word1 (str): first word
            word2 (str): second word

        Returns:
            sim (float): cosine similarity (nan if either word is not in the model)
        """
        # Compute similarity
        if word1 in self.words and word2 in self.words:
            i1, i2 = self.words.index(word1), self.words.index(word2)
            v1, v2 = self.vectors[i1], self.vectors[i2]
            sim = np.dot(v1, v2)
        else:
            sim = float("nan")
        return sim

    def get_lexicon(self, topic, top_n=10000, including_topic=True):
        """
        Get "lexicon" of words most related to a topic

        Args:
            topic (str): topic word
            top_n (int): size of lexicon
            including_topic (bool): should the topic word be included in the lexicon?

        Returns:
            lexicon (list of str): list of words most related to the topic
        """
        # Get lexicon of words most related to <topic>

        # First compute similarities (faster than constructing new matrix not including topic)
        topic_vector = self.vectors[self.words.index(topic)]
        similarities = np.matmul(self.vectors, topic_vector)
        # Sort by similarity
        sort_idx = np.argsort(similarities)
        sorted_words = [self.words[i] for i in sort_idx]
        # Remove topic word itself?
        if not including_topic:
            sorted_words.pop(sorted_words.index(topic))
        # Pare down
        lexicon = sorted_words[-top_n:]
        return lexicon

    def as_graph(self, threshold, words=None):
        """
        Convert vector model to network model

        Args:
            threshold (float): only pairs of words whose cosine similarity is greater than or equal to this threshold will share an edge in the resulting network
            words (list of str): for speed, only this subset of words will be used to produce the network (rather than all words in the vector-based model)

        Returns:
            model (NetworkModel): graph-based model
        """

        # Get only those tokens that are actually in current dictionary
        if words != None:
            words = [w for w in words if w in self.words]
        else:
            words = self.words
        pairs = get_pairs(words)
        graph = nx.Graph()
        edges = []
        for word1, word2 in pairs:
            sim = self.compute_sim(word1, word2)
            if sim >= threshold:
                graph.add_edge(word1, word2, strength=sim)
        # Create network model
        return NetworkModel(graph)


class NetworkModel(Model):
    """
    Network-based model

    Attributes:
        graph (networkx.Graph): graph of words
    """

    def __init__(self, graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected a networkx.Graph, got %s" % type(graph).__name__)
        for u, v, data in graph.edges(data=True):
            if "strength" not in data:
                raise ValueError(
                    f"Edge (%s, %s) is missing 'strength' attribute" % (u, v)
                )
        # Compute inverse strength
        inv_strength = {
            (a, b): 1 / data["strength"] for a, b, data in graph.edges(data=True)
        }
        nx.set_edge_attributes(graph, inv_strength, "inv_strength")
        self.graph = graph

    def __contains__(self, word):
        return word in self.graph

    def in_model(self, word):
        return word in self.graph

    def compute_sim(self, word1, word2):
        """
        Compute efficiency-based similarity (i.e., the length of the shortest path between words)

        Args:
            word1 (str): first word
            word2 (str): second word

        Returns:
            efficiency (float): efficiency-based similarity measure
        """
        # Compute similarity by local efficiency metric

        if word1 in self.graph and word2 in self.graph:
            try:
                distance, path = nx.bidirectional_dijkstra(
                    self.graph, word1, word2, weight="inv_strength"
                )
                efficiency = 1 / distance
            except:
                # No path between nodes
                efficiency = 0
        else:
            efficiency = float("nan")
        return efficiency

    def get_lexicon(self, topic, max_steps=2, including_topic=True):
        """
        Get "lexicon" of words most related to a topic. This function is a wrapper around networkx.ego_graph()

        Args:
            topic (str): topic word
            max_steps (int): number of steps to traverse to identify related words
            including_topic (bool): should the topic word be included in the lexicon?

        Returns:
            lexicon (list of str): list of words most related to the topic
        """
        ego_graph = nx.ego_graph(
            self.graph, n=topic, radius=max_steps, center=including_topic, distance=None
        )  # Make sure this binarizes the strengths
        lexicon = [w for w in ego_graph]
        return lexicon

    def largest_component(self, words):
        """
        Get largest network component in the subgraph induced by words

        Args:
            words (list of str): words by which to induce subgraph

        Returns:
            component (networkx.Graph): largest network component
        """
        subgraph = self.graph.subgraph(words)
        components = nx.connected_components(subgraph)
        components_by_size = list(sorted(components, key=len, reverse=True))
        if len(components_by_size) == 0:
            # No component
            largest_component = nx.Graph()
        else:
            # Get words in largest component
            largest_component = components_by_size[0]
        return largest_component


class Tokenizer:
    """
    SpaCy-based tokenizer

    Attributes:
        nlp (spacy model): SpaCy model used to tokenize
    """

    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def _lemmatize_token(self, token):
        return token.lemma_.lower()

    def _lemmatize(self, text):
        doc = self.nlp(text)
        return [self._lemmatize_token(tok) for tok in doc]

    def _is_content_token(self, tok):
        return not tok.is_stop and tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV")

    def tokenize(self, text):
        """
        Tokenize text (lowercase and keep only non-stop content words)

        Args:
            text (str): text to be tokenized

        Returns:
            tokens (list of str): list of tokens
        """
        doc = self.nlp(text)
        tokenized = [tok.text.lower() for tok in doc if self._is_content_token(tok)]
        return tokenized


def schematicity(words, model, method, topic=None, pairs=None, lexsize=None):
    """
    Compute schematicity

    Args:
        words (list of str): tokens from a narrative
        model (VectorModel or NetworkModel): model to use for computing schematicity
        method (str): method of computing schematicity ('on-topic-ppn', 'topic-relatedness', 'pairwise-relatedness', or 'component-size')
        topic (str): topic word for topic-based methods
        pairs (str): for pairwise-relatedness, which pairs should be used ('all' for all pairs, 'adj' for bigrams/adjacent pairs)
        lexsize (int): for on-topic-ppn, this parameter is passed to the .get_lexicon() method of the model

    Returns:
        schem (float): schematicity measure
    """

    # Validation
    if type(words) is not list:
        raise ValueError("words must be a list")
    if len(words) == 0:
        raise ValueError("words is empty")
    if not all(type(word) is str for word in words):
        raise ValueError("all words must be strings")
    valid_methods = [
        "on-topic-ppn",
        "topic-relatedness",
        "pairwise-relatedness",
        "component-size",
    ]
    if method not in valid_methods:
        raise ValueError("method must be one of %s" % valid_methods)

    if method in ["on-topic-ppn", "topic-relatedness"]:
        if topic == None:
            raise ValueError('topic must be specified for method "%s"' % method)
        elif topic not in model:
            raise ValueError('topic "%s" is not in model' % topic)
    elif method == "pairwise-relatedness":
        if pairs not in ["all", "adj"]:
            raise ValueError(
                'pairs must be one of "all", "adj" for method "pairwise-relatedness"'
            )
    elif method == "component-size":
        if not isinstance(model, NetworkModel):
            raise ValueError('model must be a NetworkModel for method "component-size"')

    if not all(word in model for word in words):
        raise ValueError(
            "not all words are in model. Use .keep_known() to filter out words not in the model"
        )

    if method == "on-topic-ppn":
        if isinstance(model, VectorModel):
            kwargs = {} if lexsize == None else {"top_n": lexsize}
        elif isinstance(model, NetworkModel):
            kwargs = {} if lexsize == None else {"max_steps": lexsize}
        else:
            raise ValueError("model must be a VectorModel or NetworkModel")
        lexicon = model.get_lexicon(topic, **kwargs)
        on_topic = [t in lexicon for t in words]
        ppn_on_topic = np.mean(on_topic)
        return ppn_on_topic

    elif method == "topic-relatedness":
        sims = [model.compute_sim(word, topic) for word in words]
        return np.mean(sims)

    elif method == "pairwise-relatedness":
        # Get word pairs
        if pairs == "all":
            word_pairs = get_pairs(words)
        elif pairs in ["adj", "adjacent"]:
            word_pairs = list(zip(words[:-1], words[1:]))
        else:
            raise ValueError('unrecognized pairs option "%s"' % pairs)
        # Compute average pairwise similarity
        sims = [model.compute_sim(*word_pair) for word_pair in word_pairs]
        return np.mean(sims)

    elif method == "component-size":
        # Get largest fully-connected component
        largest_component = model.largest_component(words)
        return len(largest_component) / len(words)
