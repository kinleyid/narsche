import pytest
import narsche
import os


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
    vector_mod.save(os.path.join(cur_dir, 'vector-model.mod'))


def test_load_vector_model(cur_dir):
    narsche.VectorModel.load(os.path.join(cur_dir, 'vector-model.mod'))


def test_vector_model_methods(vector_mod):
    assert vector_mod.in_model('lamp')
    vector_mod.compute_sim('lamp', 'desk')
    vector_mod.get_lexicon('lamp', top_n=2, including_topic=True)


def test_as_graph(network_mod):
    assert isinstance(network_mod, narsche.NetworkModel)


def test_network_model_methods(network_mod):
    assert network_mod.in_model('lamp')
    network_mod.compute_sim('lamp', 'desk')
    network_mod.get_lexicon('lamp', max_steps=1, including_topic=True)
    network_mod.largest_component(['lamp', 'desk'])