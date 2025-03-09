# ScratchGPT

I dived into and coded a custom implementation of a **~10.6 million parameter GPT model built from scratch** for understanding transformer architecture fundamentals. This project creates a language model trained on text data that can generate continuous text in the style of its training.

With resource help, following the paper pioneering GPT architectures and the videos of 3Blue1Brown and Andrej Karpathy, this project was an amazing opportunity to understand, recode and master the architecture of a GPT Autoregressive architecture:
- tokenization
- embedding
- multi-head self-attention mechanism
- feed-forward neural networks
- autoregressive generation


The text generated still makes approximate sense, due mostly to the fact that it implements a character level tokenization.
I am planning on recoding and implementing a sub-word level tokenization (by recoding the Byte Pair Encoding algorithm) into the model.




## Architecture

The model implements the core transformer architecture as described in "Attention Is All You Need":

![Transformer Architecture](featured/architecture.png)


### Coded and Trained Model Parameters

| **Parameter** | **Name** | **Value** | **Description** |
|-----------|-------|-------|-------------|
| **Vocabulary Size**| vocab_size | ~106 | Number of unique tokens |
| **Embedding Dimension** | n_embd | 384 | Size of token embeddings |
| **Context Length** | block_size | 256 | Maximum sequence length |
| **Batch Size** | batch_size | 64 | Samples processed per iteration |
| **Attention Heads** | n_head | 6 | Each head is 64-dimensional (384/6) |
| **Transformer Layers** | n_layer | 6 | Number of transformer blocks |

**Total Parameters: ~10.8 million**!

Parameter calculation:
- **Token embedding**: vocab_size × n_embd ≈ 106 × 384 ≈ 40 704
- **Position embedding**: block_size × n_embd = 256 × 384 ≈ 98 304
- **Self Attention**: 4 *(K,Q,V,O)* × n_layer × n_embd² ≈ 3 538 944
- **Feed Forward MLP**: n_layer × (8 × n_embd × n_embd + 4 × n_embd) ≈ 7 087 104
- **Layer Norm**: n_layer × 2 × n_embd ≈ 4 609
- **Output Layer**: n_embd × vocab_size ≈ 40 704


<br>

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
├── src/
│   ├── main.py
│   ├── model.py
│   ├── data.py
│   ├── trainer.py
│   ├── generate.py
│   └── config.py
├── weights/
├── outputs/
├── media/
│   └── shakespeare.txt
├── featured/
│   ├── architecture.png
│   ├── loss_plot.png
│   └── training.png
├── README.md
├── requirements.txt
└── .gitignore
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