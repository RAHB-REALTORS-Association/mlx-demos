import gradio as gr
import json
import math
import numpy as np
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import time
import argparse
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten

# Add your imports for the models and tokenizer here
from models import ModelArgs, Model


def build_parser():
    parser = argparse.ArgumentParser(
        description="LoRA finetuning with Llama or Mistral"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="A path to the model files containing the tokenizer, weights, config.",
    )
    # Generation args
    parser.add_argument(
        "--num-tokens", "-n", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Tokenizer:
    def __init__(self, model_path: str):
        # Initialize the tokenizer
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = " "
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        # Encode the input text
        toks = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            toks.append(self.eos_id)
        return toks

    @property
    def eos_id(self) -> int:
        # Return the end-of-sentence token ID
        return self._model.eos_id()

    def decode(self, t: List[int]) -> str:
        # Decode the token list to text
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

    @property
    def vocab_size(self) -> int:
        # Return the vocabulary size
        return self._model.vocab_size()


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        # Initialize the dataset
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        # Return the item at the given index
        return self._data[idx][self._key]

    def __len__(self):
        # Return the length of the dataset
        return len(self._data)


def load(args):
    # Load datasets and return them as train, valid, test
    names = ("train", "valid", "test")
    train, valid, test = (Dataset(Path(args.data) / f"{n}.jsonl") for n in names)
    if args.train and len(train) == 0:
        raise ValueError("Training set not found or empty.")
    if args.train and len(valid) == 0:
        raise ValueError("Validation set not found or empty.")
    if args.test and len(test) == 0:
        raise ValueError("Test set not found or empty.")
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Define the loss function
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Iterate over the batches of the dataset
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch = [
                tokenizer.encode(dset[indices[i + j]], eos=True)
                for j in range(batch_size)
            ]
            lengths = [len(x) for x in batch]

            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    # Evaluate the model on the given dataset
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    # Train the model
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()


def generate(model, prompt, tokenizer, args):
    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(args.num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)


def load_model(folder: str, dtype=mx.float16):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        model_args = ModelArgs(**config)
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = tokenizer.vocab_size
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model = Model(model_args)
    model.update(weights)
    return model, tokenizer


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load_model(args.model)

    print("Loading datasets")
    train_set, valid_set, test_set = load(args)

    if args.train:
        print("Training")
        opt = optim.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the LoRA adapter weights which we assume should exist by this point
    model.load_weights(args.adapter_file)

    if args.test:
        print("Testing")

        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        generate(model, args.prompt, tokenizer, args)

# Define the Gradio interface here
def finetune_lora(args):
    # Load pretrained model and tokenizer
    model, tokenizer = load_model(args.pretrained_model)

    if args.train:
        # Load datasets
        train_set, valid_set, _ = load(args)

        # Initialize optimizer
        optimizer = mx.optimizers.Adam(learning_rate=args.learning_rate)

        # Train model
        train(model, train_set, valid_set, optimizer, loss, tokenizer, args)

        # Save adapter weights
        mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # Load the finetuned LoRA adapter weights
    model.load_weights(args.adapter_file)

    if args.test:
        # Load test set
        _, _, test_set = load(args)

        # Evaluate the finetuned model on the test set
        test_loss = evaluate(
            model, test_set, loss, tokenizer, args.batch_size, num_batches=args.test_batches
        )
        print(f"Test loss: {test_loss:.3f}")

    if args.prompt is not None:
        # Generate text using the finetuned model
        generated_text = generate(model, args.prompt, tokenizer, args)
        print("Generated Text:\n", generated_text)

iface = gr.Interface(
    fn=finetune_lora,  # Replace with your function name
    inputs=gr.inputs.Dict(
        {
            "pretrained_model": gr.inputs.Textbox(lines=1, placeholder="Pretrained model path"),
            "train": gr.inputs.Checkbox(value=True),
            "data": gr.inputs.Textbox(lines=1, placeholder="Path to jsonl files"),
            "lora_layers": gr.inputs.Number(value=16, minimum=0, maximum=128),
            "batch_size": gr.inputs.Number(value=4, minimum=1, maximum=32),
            "iters": gr.inputs.Number(value=1000, minimum=1, maximum=100000),
            "val_batches": gr.inputs.Number(value=25, minimum=-1, maximum=1000),
            "learning_rate": gr.inputs.Number(value=1e-5, minimum=1e-6, maximum=1e-3),
            "steps_per_report": gr.inputs.Number(value=10, minimum=1, maximum=1000),
            "steps_per_eval": gr.inputs.Number(value=200, minimum=1, maximum=1000),
            "resume_adapter_file": gr.inputs.Textbox(lines=1, placeholder="Resume adapter file"),
            "adapter_file": gr.inputs.Textbox(lines=1, placeholder="Adapter file path"),
            "test": gr.inputs.Checkbox(value=False),
            "test_batches": gr.inputs.Number(value=500, minimum=-1, maximum=1000),
            "prompt": gr.inputs.Textbox(lines=1, placeholder="Prompt for text generation"),
        }
    ),
    outputs="text",  # Replace with your output type
)

iface.launch()
