import narsche
import os


def test_topic_identification():
    narsche.identify_topic(["chair", "sofa", "living", "room", "wall", "picture"])


def test_read_vectors():
    test_dir = os.path.dirname(__file__)
    sample_vec_file = os.path.join(test_dir, "sample-vectors.txt")
    mod = narsche.read_vectors(sample_vec_file)
