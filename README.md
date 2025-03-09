# ScratchGPT

I coded a custom implementation of a ~10.6 million parameter GPT model built from scratch for understanding transformer architecture fundamentals. This project creates a language model trained on text data that can generate, continuous text.

This project was an amazing opportunity to understand, recode and master the architecture of a GPT Autoregressive architecture:
- tokenization
- embedding
- multi-head self-attention mechanism
- feed-forward neural networks
- autoregressive generation




## Architecture

The model implements the core transformer architecture as described in "Attention Is All You Need":

![Transformer Architecture](featured/architecture.png)


### Coded and Trained Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary Size | ~106 | Number of unique tokens |
| Embedding Dimension | 384 | Size of token embeddings |
| Context Length | 256 | Maximum sequence length |
| Batch Size | 64 | Samples processed per iteration |
| Attention Heads | 6 | Each head is 64-dimensional (384/6) |
| Transformer Layers | 6 | Number of transformer blocks |

**Total Parameters: ~10.6 million**

Parameter calculation:
- Token embedding: vocab_size × n_embd ≈ 106 × 384
- Position embedding: block_size × n_embd = 256 × 384
- Each transformer block: 4 × n_embd² + 8 × n_embd² = 12 × n_embd²
- Output projection: n_embd × vocab_size ≈ 384 × 106



## Training Performance

The model was trained on Ecole Polytechnique's SSH GPU infrastructure with impressive results:

- Training time: ~23 minutes
- Processing speed: 5.33 iterations/second
- Final training loss: 1.0992
- Final validation loss: 1.4863

![Loss Evolution](featured/loss_plot.png)


## Sample Generated Text

The model generates Shakespeare-like text after training. Here's a brief sample:

```
VIRGILIA:
And this is in Ruvers.

VIRGILIA:
Let him go with, or his desire, his grace these
retire, in my mind give a son: which is for
you known anoted his nature; and, is it is
so make, people, amissive; and, good state
with to complex them.

MARIANA:
And as he, Clarence, let him to endure his wills,
Again one do never kiss
Do wrong to curse. Take me him foundly; God's fall.

GRUMIO:
How sociation like to bear.

SAMILLO:
In that may great them delivers;
The washest I think his grace backs, was death
My patrived; the sung it of haven can it witness
To part care them, and being clears the orpeasies
Will these against the third activern to piece
To common, true them on:
Tellion him he our suffices himself him friends,
And all thy wondless years
```



## Usage and running


### Repo Structure

```
ScratchGPT/
│
├── main.py              # Main script to run training/generation
├── model.py             # Model architecture
├── data.py              # Data loading and preprocessing
├── trainer.py           # Training functions
├── generate.py          # Text generation functions
├── config.py            # Configuration parameters
│
├── weights/             # Directory for saved model weights
│   └── weights.pth        # Trained model weights
│
├── outputs/             # Directory for generated text
│   └── output.txt       # Generated text output
│
├── loss.csv             # Training loss history
│
├── featured/
│   ├── architecture.png # Transformer architecture diagram
│   ├── loss_plot.png    # Loss evolution graph
│   └── training.png     # Training screenshot
│
└── media/
    └── shakespeare.txt  # Training data
```

### Training the Model

If you want to train the model:

```bash
python main.py --train --weights model.pth
```

### Generating Text

If you want to generate text from a loaded weights trained model:

```bash
python main.py --weights model.pth --output output.txt
```

### Other Usage

You can customize the model's behavior in the `config.py` file.




## License

[MIT License](LICENSE)

## Acknowledgements

This project draws inspiration from the transformer architecture described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and the GPT model series by OpenAI. Additional learning resources include Andrej Karpathy's video tutorials and 3Blue1Brown explanations.