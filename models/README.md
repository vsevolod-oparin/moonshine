# models/

Moonshine model architecture files copied from `transformers.models.moonshine`.

These files use `transformers`-internal imports (e.g. `from ...activations import ACT2FN`)
and are **not yet adapted for standalone use**. They serve as the starting reference
for architecture implementation in **M4: Model Architecture**.

To load the model via transformers (for reference/inspection):
```python
from transformers import MoonshineForConditionalGeneration, MoonshineConfig
```
