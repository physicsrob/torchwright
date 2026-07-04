# Ops Module

## Layout: the import path is the machine choice

- `torchwright.ops.relu` — the frozen ReLU-machine op library.
- `torchwright.ops.swiglu` — the swish-machine op library.
- Machine-neutral code lives at this level: `const`, `inout_nodes`,
  `linear` (purely linear ops — no MLP sublayer, so no activation
  choice), `attention_ops` (attention hardware), `_math` (private
  shared helpers).

There are no re-exports at this level and no mode flags — importing
from a machine package *is* the machine choice, and the compiler's
uniformity check (all FFNs in a graph share one activation) is the
backstop against accidental mixing.

## Recommended import style

Import from the specific module that owns the op:

```python
from torchwright.ops.linear import add_const, subtract
from torchwright.ops.swiglu.arithmetic_ops import compare
from torchwright.ops.swiglu.arithmetic_ops import min as min_node
```

Ops that shadow Python builtins (`abs`, `min`) get an `as` alias at the
call site, as in the last line.

## Operation Naming Convention

Operation Naming Convention:
- Operation names start with a verb where possible. Most operations are performing actions on nodes, and verbs make it easier to interpret what the operation will do.
- Use the term `_const` suffix when the action is between a node and a constant (Python float), and `vector` when it involves a vector.
- It can be assumed that the action being performed is on one or more node.  If, however, the action is being performed between a node and a constant, state so with the `_const` suffix.  For instance `add_const` clearly indicates that the input node is being added with a constant value.
- Stick to PyTorch naming conventions where they align. If there's an analogous operation in PyTorch, its naming convention can serve as guidance, but only if it doesn't confuse the purpose of the function in this new context.
- Keep names as concise as possible without sacrificing clarity. If a name can be shorter without being less clear, it should be.
- Operations that are predicated on a boolean condition should be prefixed with `cond_`
- Operations that conditionally copy input to output should be prefixed with `select_`


Parameter naming convention:
- Boolean input nodes should always be named `cond`.
- If an operation acts on a single node input, that node should be named `inp`.
- If an operation acts on two node inputs, those nodes should be named `inp1` and `inp2`.
- If an operation has a constant (float) parameter, the parameter name should be scalar.
- If an operation has a vector parameter, the parameter name should be vector.
