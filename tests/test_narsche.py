import pytest
import narsche
import os
import numpy as np
from pdb import set_trace


@pytest.fixture
def example_words():
    return ["sitting", "lamp", "desk", "office"]


@pytest.fixture
def cur_dir():
    test_dir = os.path.dirname(__file__)
    return test_dir


@pytest.fixture
def vector_mod(cur_dir):
    sample_vec_file = os.path.join(cur_dir, "sample-vectors.txt")
    mod = narsche.read_vectors(sample_vec_file)
    # assert isinstance(vector_mod, narsche.VectorModel)
    return mod


@pytest.fixture
def network_mod(vector_mod):
    return vector_mod.as_graph(threshold=0.9)


def test_topic_identification():
    narsche.identify_topic(["chair", "sofa", "living", "room", "wall", "picture"], return_scores=True)


def test_read_vector(cur_dir):
    file = os.path.join(cur_dir, "sample-vectors.txt")
    # assert isinstance(vector_mod, narsche.VectorModel)


def test_save_vector_model(vector_mod, cur_dir):
    vector_mod.save(os.path.join(cur_dir, "vector-model.mod"))
    vector_mod.save(os.path.join(cur_dir, "vector-model"))


def test_load_vector_model(cur_dir):
    narsche.VectorModel.load(os.path.join(cur_dir, "vector-model.mod"))
    narsche.VectorModel.load(os.path.join(cur_dir, "vector-model"))


def test_vector_model_methods(vector_mod):
    assert "lamp" in vector_mod
    sim = vector_mod.compute_sim("lamp", "desk")
    assert 0.98 < sim < 1
    lex = vector_mod.get_lexicon("lamp", top_n=2, including_topic=True)
    assert set(lex) == {"desk", "lamp"}
    lex = vector_mod.get_lexicon("lamp", top_n=2, including_topic=False)
    assert set(lex) == {"desk", "chair"}
    vector_mod[('desk', 'lamp')]


def test_as_graph(vector_mod, network_mod):
    assert isinstance(network_mod, narsche.NetworkModel)
    assert len(network_mod.graph) == 4
    new_net = vector_mod.as_graph(words=["lamp", "desk", "pottery"], threshold=0.3)
    assert len(new_net.graph) == 2


def test_save_network_model(network_mod, cur_dir):
    network_mod.save(os.path.join(cur_dir, "network-model.mod"))


def test_load_network_model(cur_dir):
    narsche.NetworkModel.load(os.path.join(cur_dir, "network-model.mod"))


def test_network_model_methods(network_mod):
    assert "lamp" in network_mod
    network_mod.compute_sim("lamp", "desk")
    network_mod.get_lexicon("lamp", max_steps=1, including_topic=True)
    network_mod.get_lexicon("lamp", max_steps=1, including_topic=False)
    network_mod.largest_component(["lamp", "desk"])


def test_tokenizer():
    tokenizer = narsche.Tokenizer()
    tokenizer.tokenize("This is a short piece of text", lemmatize=True)


def test_schematicity_vector_model(vector_mod, example_words):
    words = vector_mod.keep_known(example_words)
    narsche.schematicity(
        model=vector_mod, words=words, method="on-topic-ppn", topic="lamp", lex_size=2
    )
    narsche.schematicity(
        model=vector_mod, words=words, method="topic-relatedness", topic="lamp"
    )
    narsche.schematicity(
        model=vector_mod, words=words, method="pairwise-relatedness", pairs="adj"
    )


def test_schematicity_network_model(network_mod, example_words):
    words = network_mod.keep_known(example_words)
    narsche.schematicity(
        model=network_mod, words=words, method="on-topic-ppn", topic="lamp"
    )
    narsche.schematicity(
        model=network_mod, words=words, method="topic-relatedness", topic="lamp"
    )
    narsche.schematicity(
        model=network_mod, words=words, method="pairwise-relatedness", pairs="adj"
    )
    narsche.schematicity(model=network_mod, words=words, method="component-size")

def test_vec_errors(vector_mod):
    with pytest.raises(Exception):
        narsche.VectorModel(words='hey', vectors=[1,2,3])
    with pytest.raises(Exception):
        narsche.VectorModel(words=['hey'], vectors=[[1,2,3]])
    with pytest.raises(Exception):
        narsche.VectorModel(words=['hey', 'hi'], vectors=np.array([[1,2,3]]))
    with pytest.raises(Exception):
        vector_mod[1]
    with pytest.raises(Exception):
        vector_mod.get_lexicon("lamp", top_n=100, including_topic=True)
    with pytest.raises(Exception):
        narsche.VectorModel.load(os.path.join(cur_dir, "network-model.mod"))

def test_net_errors():
    with pytest.raises(Exception):
        narsche.NetworkModel.load(os.path.join(cur_dir, "vector-model.mod"))

def test_schematicity_errors(vector_mod, example_words):
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words='a b c', method="on-topic-ppn", topic="lamp", lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=[], method="on-topic-ppn", topic="lamp", lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=['a', 1], method="on-topic-ppn", topic="lamp", lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words, method="my-best-method", topic="lamp", lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words, method="on-topic-ppn", lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words, method="on-topic-ppn", topic='foo', lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words, method="pairwise-relatedness", pairs='any', lex_size=2
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words, method="component-size"
        )
    with pytest.raises(Exception):
        narsche.schematicity(
            model=vector_mod, words=example_words + ['foo'], method="pairwise-relatedness", pairs='all'
        )
    