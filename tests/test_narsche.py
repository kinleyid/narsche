import pytest
import narsche
import os
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
    return mod


@pytest.fixture
def network_mod(vector_mod):
    return vector_mod.as_graph(threshold=0.9)


def test_topic_identification():
    narsche.identify_topic(["chair", "sofa", "living", "room", "wall", "picture"])


def test_read_vector(vector_mod):
    assert isinstance(vector_mod, narsche.VectorModel)


def test_save_vector_model(vector_mod, cur_dir):
    vector_mod.save(os.path.join(cur_dir, "vector-model.mod"))


def test_load_vector_model(cur_dir):
    narsche.VectorModel.load(os.path.join(cur_dir, "vector-model.mod"))


def test_vector_model_methods(vector_mod):
    assert "lamp" in vector_mod
    sim = vector_mod.compute_sim("lamp", "desk")
    assert 0.98 < sim < 1
    lex = vector_mod.get_lexicon("lamp", top_n=2, including_topic=True)
    assert set(lex) == {'desk', 'lamp'}
    vector_mod.get_lexicon("lamp", top_n=2, including_topic=False)
    assert set(lex) == {'desk', 'chair'}


def test_as_graph(vector_mod, network_mod):
    assert isinstance(network_mod, narsche.NetworkModel)
    vector_mod.as_graph(words=["lamp", "desk", "pottery"], threshold=0.3)
    assert len(network_mod.graph) == 2


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
    tokenizer.tokenize("This is a short piece of text")


def test_schematicity_vector_model(vector_mod, example_words):
    words = vector_mod.keep_known(example_words)
    narsche.schematicity(
        model=vector_mod, words=words, method="on-topic-ppn", topic="lamp"
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
