from torchwright.ops.inout_nodes import create_embedding, create_unembedding
from torchwright.ops.relu.map_select import map_to_table


def test_embedding():
    vocab = ["1", "2", "3", "was1", "was2", "was3", "default"]
    embedding = create_embedding(vocab=vocab)
    table_values = {"1": "was1", "2": "was2", "3": "was3"}
    lookup = map_to_table(
        inp=embedding,
        key_to_value={
            embedding.get_embedding(text): embedding.get_embedding(value)
            for text, value in table_values.items()
        },
        default=embedding.get_embedding("default"),
    )

    output_node = create_unembedding(lookup, embedding)
    output = output_node.compute(
        n_pos=6,
        input_values={"embedding_input": ["1", "2", "3", "2", "1", "10"]},
    )
    assert output == ["was1", "was2", "was3", "was2", "was1", "default"]
