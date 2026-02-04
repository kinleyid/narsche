import pickle
import numpy as np
from wordfreq import word_frequency
from collections import Counter
import networkx as nx
import spacy
import sys
import gzip
from pdb import set_trace

epsilon = sys.float_info.epsilon


def get_pairs(iterable):
    """
    Gets adjacent pairs of items from some iterable collection.

    Parameters
    ----------
    iterable : list, tuple, or other iterable
        Come iterable collection such as a list or tuple.

    Returns
    -------
    pairs : list
        Adjacent pairs or items in a list.
    """

    pairs = [(a, b) for i, a in enumerate(iterable) for b in iterable[(i + 1) :]]
    return pairs


def identify_topic(words, return_scores=False):
    """
    Identify topic using tf-idf

    Parameters
    ----------
    words : list
        A list of words in a narrative. Note that because term frequency is used to evaluate topic candidacy, this should not be a list of only the unique words in a narrative---it a word occurs multiple times in the narrative, it should occur multiple times in this list.
    return_scores: bool, optional
        Whether to return the TF-IDF scores for each word in a dictionary in addition to the topic. The default is False.

    Returns
    -------
    topic : str
        The topic word.
    scores : dict
        TF-IDF scores for each word.
    """

    bag = set(words)  # unique words
    # get term frequency
    tf = Counter(words)
    # compute doc frequency (could be 0, in which case sub in epsilon)
    df = {word: word_frequency(word=word, lang="en", minimum=epsilon) for word in bag}
    idf = {word: np.log(1 / df[word]) for word in bag}
    tf_idf = {word: tf[word] * idf[word] for word in bag}
    # sort ascendingly
    tf_idf = {k: v for k, v in sorted(tf_idf.items(), key=lambda item: -item[1])}
    # topic is first item
    topic = next(iter(tf_idf))
    assert tf_idf[topic] == max(
        tf_idf.values()
    )  # probably not necessary but peace of mind
    if return_scores:
        return topic, tf_idf
    else:
        return topic


def read_vectors(file, encoding="utf-8", normalize=True, archive=False):
    """
    Create vector model from text file containing word vectors.

    Parameters
    ----------
    file : str
        Path to text file containing word vectors.
    encoding : str, optional
        The encoding to use when reading the text file. The default is "utf-8".
    normalize : bool, optional
        Specifies whether to normalize the vectors for cosine similarity computation. Can be skipped for speed if vectors are already normalized. The default is True.
    archive : bool, optional
        Specifies whether the text files are in an archive format (e.g. .gz, .zip). The default is False.

    Returns
    -------
    model : VectorModel
        A vector model based on the embeddings in the text file.
    """

    words = []
    vectors = []
    if archive:
        open_fn = lambda file: gzip.open(file, "rt", encoding=encoding)
    else:
        open_fn = lambda file: open(file, "r", encoding=encoding)
    with open_fn(file) as f:
        for line in f:
            # First item in space-delimited line is token, remaining items are vector elements
            split_line = line.rstrip("\n").split(" ")
            words.append(split_line[0])
            # Normalize vector for fast dot product-based cosine similarity computation
            vector = np.asarray(split_line[1:]).astype(np.float32)
            # vector /= np.linalg.norm(vector)
            vectors.append(vector)
    vectors = np.array(vectors)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= norms
    model = VectorModel(words, vectors)
    return model


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
                "Expected instance of %s, got %s" % (cls.__name__, type(obj).__name__)
            )
        return obj

    def keep_known(self, words):
        return [word for word in words if word in self]


class VectorModel(Model):
    """
    Vector-based model

    Attributes
    ----------
    words : dict
        Dictionary mapping words in model to indices of word vectors in the matrix of word vectors (see below).
    vectors : np.ndarray
        Matrix whose rows are the word vectors.

    Methods
    -------
    compute_sim(word1, word2)
        Compute cosine similarity between word pairs.
    get_lexicon(topic, top_n=10000, including_topic=True)
        Get a lexicon of the words most related to a given topic.
    as_graph(threshold, words=None)
        Construct a NetworkModel with thresholded connections based on cosine similarity
    """

    def __init__(self, words, vectors):
        """
        Parameters
        ----------
        words : list of strings
            A list of words corresponding to the word embeddings in "vectors".
        vectors : np.ndarray
            A matrix whose rows are the word embeddings.
        """

        if isinstance(words, list):
            if not all(isinstance(word, str) for word in words):
                raise ValueError("words is not a list of strings")
        if not isinstance(vectors, np.ndarray):
            raise ValueError("vectors is not an np.ndarray")
        if len(words) != len(vectors):
            raise ValueError("different numbers of words and vectors")
        # Store as dict of words and matrix of vectors
        self.words = {w: i for i, w in enumerate(words)}
        self.vectors = vectors

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self.words[key]
            return self.vectors[idx]
        if isinstance(key, (list, set, tuple)):
            idx = [self.words[w] for w in key]
            return self.vectors[idx]
        else:
            raise ValueError("key must be a string or iterable of strings")

    def __contains__(self, word):
        return word in self.words

    def compute_sim(self, word1, word2):
        """
        Compute cosine similarity between words

        Parameters
        ----------
        word1 : str
            First word.
        word2 : str
            Second word.

        Returns
        -------
        sim : float
            Cosine similarity (nan if either word is not in the model).
        """
        # Compute similarity
        if word1 in self.words and word2 in self.words:
            sim = np.dot(self[word1], self[word2])
        else:
            sim = float("nan")
        return sim

    def get_lexicon(self, topic, top_n=10000, including_topic=True):
        """
        Get "lexicon" of words most cosine-similar to a topic word

        Parameters
        ----------
        topic : str
            Topic word.
        top_n : int, optional
            Size of lexicon to extract. Default is 10,000
        including_topic : bool, optional
            Specifies whether the topic word should be included in the lexicon. Default is True.

        Returns
        -------
        lexicon : list of strings
            List of words most related to the topic word.
        """

        # First compute similarities (faster than constructing new matrix not including topic)
        similarities = np.matmul(self.vectors, self[topic])
        if not including_topic:
            # Topic word is guaranteed to be most similar to itself
            top_n = top_n + 1
        # Split at top_n
        lex_idx = np.argpartition(similarities, -top_n)[-top_n:]
        # Need to map from index to words---only create this inverse mapping once as needed
        if "_rev_idx" not in dir(self):
            self._rev_idx = {i: w for w, i in self.words.items()}
        lexicon = [self._rev_idx[i] for i in lex_idx]
        # Remove topic word itself?
        if not including_topic:
            lexicon.pop(lexicon.index(topic))
        return lexicon

    def as_graph(self, threshold, words=None):
        """
        Convert vector model to network model.

        Parameters
        ----------
        threshold : float
            Only pairs of words whose cosine similarity is greater than or equal to this threshold will share an edge in the resulting network.
        words : list of strings, optional
            For speed, only this subset of words will be used to produce the network (rather than all words in the vector-based model).

        Returns
        -------
        model : NetworkModel
            Graph-based model.
        """

        # Get only those tokens that are actually in current dictionary
        if words == None:
            words = self.words
        pairs = get_pairs(words)
        graph = nx.Graph()
        for word1, word2 in pairs:
            sim = self.compute_sim(word1, word2)
            if sim >= threshold:
                graph.add_edge(word1, word2, weight=sim)
        # Create network model
        return NetworkModel(graph)


class NetworkModel(Model):
    """
    Network-based model

    Attributes
    ----------
    graph : networkx.Graph
        Graph of words and their weighted connections

    Methods
    -------
    compute_sim(word1, word2)
        Compute similarity by length of shortest path between words.
    get_lexicon(topic, max_steps=2, including_topic=True)
        Get "lexicon" of words most related to a topic word (its ego graph).
    largest_component(words)
        Get largest component on the subgraph induced by a set of words.
    """

    def __init__(self, graph, compute_inverse_weight=True):
        """
        Parameters
        ----------
        graph : networkx.Graph
            Network of words whose edges include a "weight" attribute.
        compute_inverse_weight : bool, optional
            Specifies whether to compute an additional "inverse_weight" attribute, which is used to find the shortest path length between words. Default is True.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Expected a networkx.Graph, got %s" % type(graph).__name__)

        if compute_inverse_weight:
            # Compute inverse weight
            inv_weight = {
                (a, b): 1 / data["weight"] for a, b, data in graph.edges(data=True)
            }
            nx.set_edge_attributes(graph, inv_weight, "inv_weight")

        self.graph = graph

    def __contains__(self, word):
        return word in self.graph

    def compute_sim(self, word1, word2):
        """
        Compute efficiency-based similarity (i.e., the length of the shortest path between words).

        Parameters
        ----------
        word1 : str
            First word.
        word2 : str
            Second word

        Returns
        -------
        efficiency : float
            Efficiency-based similarity measure (0 if no path exists between words; nan if either word is not in the graph).
        """
        # Compute similarity by local efficiency metric

        if word1 in self.graph and word2 in self.graph:
            try:
                distance, path = nx.bidirectional_dijkstra(
                    self.graph, word1, word2, weight="inv_weight"
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

        Parameters
        ----------
        topic : str
            Topic word.
        max_steps : int, optional
            Number of steps to traverse to identify related words. Default is 2.
        including_topic : bool, optional
            Specifies where the topic word should be included in the lexicon. Default is True.

        Returns:
        lexicon : list of strings
            List of words most related to the topic.
        """
        ego_graph = nx.ego_graph(
            self.graph, n=topic, radius=max_steps, center=including_topic, distance=None
        )
        lexicon = [w for w in ego_graph]
        return lexicon

    def largest_component(self, words):
        """
        Get largest network component in the subgraph induced by words. The "component-size" schematicity measure is the cardinality of this largest component divided by the cardinality of the subgraph representing the entire narrative.

        Parameters
        ----------
        words : list of strings
            Words by which to induce subgraph.

        Returns
        -------
        component : networkx.Graph
            Largest network component.
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

    Attributes
    ----------
    nlp : spacy model
        SpaCy model used to tokenize

    Methods
    -------
    tokenize(text, lowercase=True, rm_stops=True, only_content=True, lemmatize=False)
        Tokenize a piece of text.
    """

    def __init__(self, spacy_model="en_core_web_sm"):
        """
        Parameters
        ----------
        spacy_model : str, optional
            Spacy model to use for tokenization. The default is "en_core_web_sm".
        """

        self.nlp = spacy.load(spacy_model)

    def _lemmatize_token(self, token):
        return token.lemma_.lower()

    def _lemmatize(self, text):
        doc = self.nlp(text)
        return [self._lemmatize_token(tok) for tok in doc]

    def _is_content(self, tok):
        return tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV")

    def tokenize(
        self,
        text,
        lowercase=True,
        rm_stops=True,
        only_content=True,
        lemmatize=False,
    ):
        """
        Tokenize text (lowercase and keep only non-stop content words)

        Parameters
        ----------
        text : str
            Text to be tokenized.
        lowercase : bool, optional
            Specifies whether to convert words to lowercase. Default is True.
        rm_stops : bool, optional
            Specifies whether to remove stopwords. Default is True.
        only_content : bool, optional
            Specifies whether to keep only content words (nounds, verbs, adjectives, adverbs) as identified by SpaCy's part-of-speech tagger. Default is True.
        lemmatize : bool, optional
            Specifies whether to lemmatize tokens using SpaCy's lemmatizer. Default is False.

        Returns
        -------
        tokens : list of strings
            List of tokens
        """
        doc = self.nlp(text)
        tokenized = []
        for tok in doc:

            keep_word = True

            if rm_stops and tok.is_stop:
                keep_word = False

            if only_content and tok.pos_ not in ("NOUN", "VERB", "ADJ", "ADV"):
                keep_word = False

            if keep_word:

                if lemmatize:
                    word = tok.lemma_
                else:
                    word = tok.text

                if lowercase:
                    word = word.lower()

                tokenized.append(word)

        return tokenized


def schematicity(words, model, method, topic=None, pairs=None, lexsize=None):
    """
    Compute schematicity using a variety of methods

    Parameters
    ----------
    words : list of strings
        Tokens from a narrative.
    model : VectorModel or NetworkModel
        The model to use for computing schematicity.
    method : str
        The method of computing schematicity ('on-topic-ppn', 'topic-relatedness', 'pairwise-relatedness', or 'component-size').
    topic : str
        Topic word to use for topic-based methods (on-topic-ppn and topic-relatedness). Ignored for other methods.
    pairs : str
        For the pairwise-relatedness measure, which pairs should be used ('all' for all pairs, 'adj' for bigrams/adjacent pairs). Ignored for other methods.
    lexsize : int
        For on-topic-ppn, this parameter is passed to the .get_lexicon() method of the model. Ignored for other methods.

    Returns
    -------
    schematicity : float
        Schematicity measure computed by the requested method.
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
