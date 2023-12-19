# Stable Diffusion Demo

## Introduction

Welcome to the Stable Diffusion demo in the MLX Demos repository! This demonstration showcases the capabilities of MLX for image generation using Stable Diffusion. Here, you can experience how textual prompts can be transformed into stunning visual representations.

## Installation

To get started with the Stable Diffusion demo, you need to have Python installed on your system along with a few dependencies.

### Prerequisites

- Python 3.6 or higher
- Git (for cloning this repository)

### Steps

1. **Clone the Repository:**

   If you haven't already cloned the entire MLX Demos repository, you can do so by running:

   ```bash
   git clone https://github.com/RAHB-REALTORS-Association/mlx-demos.git
   cd mlx-demos/stable-diffusion
   ```

2. **Install Dependencies:**

   Inside the `stable-diffusion` directory, install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install packages like `mlx`, `safetensors`, `huggingface-hub`, `numpy`, `gradio` and others necessary for running the demo.

## Running the Demo

To run the Stable Diffusion demo:

1. Ensure you are in the `stable-diffusion` directory.

2. Run the demo application:

   ```bash
   python app.py
   ```

3. Open your web browser and navigate to the URL provided by Gradio: `http://127.0.0.1:7860`.

4. Use the Gradio interface to input your textual prompts and generate images.

## Notes

- The image generation process may take a few moments depending on the complexity of the prompt and your system's capabilities.
- Ensure you have a stable internet connection if the model requires online resources.

Enjoy generating fascinating images using Stable Diffusion and MLX!