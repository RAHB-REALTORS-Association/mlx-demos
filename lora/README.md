# LoRA Text Generation Demo

## Introduction

Welcome to the LoRA (Low-Rank Adaptation) Text Generation demo in the MLX Demos repository! This demonstration utilizes LoRA techniques to fine-tune models for text generation tasks. It showcases the integration of MLX with text processing and model adaptation.

## Installation

To get started with the LoRA Text Generation demo, ensure you have Python installed on your system along with some necessary dependencies.

### Prerequisites

- Python 3.6 or higher
- Git (for cloning this repository)

### Steps

1. **Clone the Repository:**

   If you haven't already cloned the entire MLX Demos repository, you can do so by running:

   ```bash
   git clone https://github.com/RAHB-REALTORS-Association/mlx-demos.git
   cd mlx-demos/lora
   ```

2. **Install Dependencies:**

   Inside the `lora` directory, install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install packages like `mlx`, `sentencepiece`, `torch`, and `gradio`, which are essential for the LoRA demo.

## Running the Demo

To run the LoRA Text Generation demo:

1. Make sure you are in the `lora` directory.

2. Start the demo application:

   ```bash
   python app.py
   ```

3. Open your web browser and go to the Gradio URL: `http://127.0.0.1:7860`.

4. Use the Gradio interface to configure model parameters and input text prompts for generation.

## Notes

- Depending on the configuration and model size, the text generation process might take varying amounts of time.
- A stable internet connection is recommended if the model requires downloading resources or online interaction.

Enjoy exploring the power of LoRA in text generation with MLX!