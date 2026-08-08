import torch

from torchwright.graph import Embedding
from torchwright.ops.inout_nodes import create_onehot_embedding


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


def test_onehot_embedding_appends_zero_unk_without_shifting_ids():
    embedding = create_onehot_embedding(["a", "b", "<eos>"])

    assert embedding.tokenizer.vocab == ["a", "b", "<eos>", "<unk>"]
    assert embedding.d_embed == 3
    assert embedding.table.shape == (4, 3)
    assert torch.equal(embedding.table[:3], torch.eye(3))
    assert torch.equal(embedding.get_embedding("<unk>"), torch.zeros(3))
    assert torch.equal(embedding.get_embedding("not-in-vocab"), torch.zeros(3))


def test_onehot_embedding_uses_explicit_unk_position_as_zero_row():
    embedding = create_onehot_embedding(["a", "<unk>", "b"])

    assert embedding.tokenizer.vocab == ["a", "<unk>", "b"]
    assert embedding.d_embed == 2
    assert torch.equal(
        embedding.table,
        torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    )
