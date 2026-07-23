import torch

from torchwright.graph import Embedding


def test_embedding():
    embedding = Embedding(vocab=["1", "2", "3"])

    output1 = embedding.compute(n_pos=1, input_values={"embedding_input": "1"})
    output2 = embedding.compute(n_pos=1, input_values={"embedding_input": "2"})
    output3 = embedding.compute(n_pos=1, input_values={"embedding_input": "3"})
    output4 = embedding.compute(n_pos=1, input_values={"embedding_input": "4"})
    output5 = embedding.compute(n_pos=1, input_values={"embedding_input": "5"})

    # Vector for 1/2/3 should all be different
    assert not torch.allclose(output1, output2)
    assert not torch.allclose(output1, output3)
    assert not torch.allclose(output2, output3)

    assert torch.allclose(output1, embedding.get_embedding("1"))
    assert torch.allclose(output2, embedding.get_embedding("2"))
    assert torch.allclose(output3, embedding.get_embedding("3"))

    # Vector 4/5 should be the same (<unk>)
    assert torch.allclose(output4, output5)
    assert torch.allclose(output4, embedding.get_embedding("<unk>"))
    assert torch.allclose(output5, embedding.get_embedding("<unk>"))
