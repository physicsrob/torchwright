import torch

from torchwright.graph.node import Node
from torchwright.graph.spherical_codes import get_spherical_codes
from torchwright.graph.value_type import NodeValueType

unk_token = "<unk>"
bos_token = "<bos>"
default_special_tokens = [unk_token]

#: Rank of an integer-ID input shaped ``(n_pos, 1)`` rather than ``(n_pos,)``.
_COLUMN_VECTOR_NDIM = 2


class Tokenizer:
    vocab: list[str]

    def __init__(
        self, vocab: list[str], special_tokens: list[str] | None = None
    ) -> None:
        if special_tokens is None:
            special_tokens = default_special_tokens
        self.vocab = list(special_tokens) + list(vocab)

    def get_token_id(self, text: str) -> int:
        if text in self.vocab:
            return self.vocab.index(text)
        return self.vocab.index(unk_token)

    def decode_id(self, token_id: int) -> str:
        if token_id < len(self.vocab):
            return self.vocab[token_id]
        return unk_token

    def __len__(self) -> int:
        return len(self.vocab)


class Embedding(Node):
    tokenizer: Tokenizer
    table: torch.Tensor
    max_vocab: int
    d_embed: int
    input_name: str

    def __init__(
        self,
        vocab: list[str],
        d_embed: int = 8,
        table: torch.Tensor | None = None,
        input_name: str = "embedding_input",
        special_tokens: list[str] | None = None,
    ) -> None:
        """Embedding lookup node.

        Default behaviour (``table=None``, ``d_embed=8``): loads the E8
        spherical-code table and prepends an ``<unk>`` row via the
        default ``special_tokens``. This matches all pre-existing
        callers.

        Caller-supplied table: pass ``table`` of shape
        ``(|special_tokens| + |vocab|, d_embed)`` and any ``d_embed``.
        For a direct row-to-vocab mapping (no ``<unk>`` prefix), pass
        ``special_tokens=[]``.

        ``input_name`` selects which ``input_values`` slot
        :meth:`compute` reads from. Legacy default ``"embedding_input"``
        is kept so existing callers continue to work.
        """
        self.d_embed = d_embed
        self.input_name = input_name
        self.tokenizer = Tokenizer(vocab, special_tokens=special_tokens)

        if table is None:
            self.table = get_spherical_codes(d_embed)
            self.max_vocab = self.table.shape[0]
        else:
            expected_rows = len(self.tokenizer)
            assert table.shape == (expected_rows, d_embed), (
                f"table shape {tuple(table.shape)} does not match "
                f"(|tokenizer vocab|, d_embed) = ({expected_rows}, {d_embed})"
            )
            self.table = table
            self.max_vocab = expected_rows

        assert self.table.shape == (self.max_vocab, d_embed)
        super().__init__(d_embed, [])

    def get_embedding(self, text: str) -> torch.Tensor:
        return self.table[self.tokenizer.get_token_id(text)]

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        assert self.input_name in input_values
        raw = input_values[self.input_name]

        if isinstance(raw, torch.Tensor):
            ids = raw
            if ids.ndim == _COLUMN_VECTOR_NDIM:
                assert ids.shape[1] == 1, (
                    f"integer-ID input must be (n,) or (n, 1); got {tuple(ids.shape)}"
                )
                ids = ids[:, 0]
            assert ids.shape == (n_pos,), (
                f"expected integer-ID input of shape ({n_pos},); got {tuple(ids.shape)}"
            )
            ids = ids.to(dtype=torch.long)
        else:
            assert len(raw) == n_pos
            ids = torch.tensor(
                [self.tokenizer.get_token_id(x) for x in raw],
                dtype=torch.long,
            )

        # Match the table's device + dtype to the caller's context.
        # Device: CPU table vs. GPU compiled module — the gather op
        # needs both on the same device.
        # Dtype: the fp64 branch of probe_compiled swaps the default
        # dtype and converts every Linear/Attn weight to float64;
        # the Embedding's table has to follow or downstream matmuls
        # raise "float != double".
        target_dtype = torch.get_default_dtype()
        table = self.table.to(device=ids.device, dtype=target_dtype)
        result = table[ids]
        assert result.shape == (n_pos, self.d_embed)
        return result

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def num_params(self) -> int:
        return self.d_embed * self.max_vocab


class Unembedding:
    def __init__(self, inp: Node, embedding: Embedding) -> None:
        self.embedding = embedding
        self.inp = inp
        assert len(self.inp) == self.embedding.d_embed

    def compute(self, n_pos: int, input_values: dict) -> list[str]:
        input_value = self.inp.compute(n_pos, input_values)
        result = []
        for pos in range(n_pos):
            probs = self.embedding.table @ input_value[pos]
            token_id = probs.argmax().item()
            assert isinstance(token_id, int)
            result.append(self.embedding.tokenizer.decode_id(token_id))
        return result
