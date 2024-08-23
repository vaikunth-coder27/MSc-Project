
# Investigating Memorization in Code-Based Large Language Models

## Project Overview

This repository contains the code and documentation for the thesis "Investigating Memorization in Code-Based Large Language Models." The research focuses on exploring the memorization tendencies of various code-based large language models (LLMs) across different architectures and programming languages, with a specific emphasis on the impact of few-shot learning, quantization, and prompt tuning.

## Directory Structure

The project directory is structured as follows:

```plaintext
PROJECT/
│
├── requirements.txt
│
├── Evaluating_Memorization_Across_LLM_Architectures/
│   ├── Non-Autoregressive-Task.py
│   ├── Autoregressive-T5.py
│   ├── Autoregressive-GPT.py
│   ├── codebleu/
│       └── codebleu.py
│
├── Few-shot/
│   ├── fsl-codeT5.py
│   ├── fsl-codeGPT.py
│
└── Prompt-Tuning/
    ├── codeT5-PT.py
    ├── codeGPT-PT.py
    ├── codeBERT-PT.py
```

## Datasets and Models

### Datasets
The dataset used for this project is CodeSearchNet, an open-source dataset specifically curated for code search and natural language processing tasks. The dataset can be accessed from Hugging Face using the following URL:
- [CodeSearchNet Dataset](https://huggingface.co/datasets/code-search-net/code_search_net)

### Models
The following open-source models were used:
- **CodeBERT**: An encoder-only model pre-trained on code data.
  - [CodeBERT on Hugging Face](https://huggingface.co/microsoft/codebert-base-mlm)
- **CodeT5**: An encoder-decoder model designed for both code understanding and generation.
  - [CodeT5 on Hugging Face](https://huggingface.co/Salesforce/codet5-large)
- **CodeGPT**: A decoder-only model fine-tuned for code generation tasks.
  - [CodeGPT on Hugging Face](https://huggingface.co/AISE-TUDelft/CodeGPT-Multilingual)

## Running the Experiments

### 1. Evaluating Memorization Across LLM Architectures
To reproduce the results from Chapter 4 of the thesis, navigate to the `Evaluating_Memorization_Across_LLM_Architectures` directory and run the following scripts:

- **Non-Autoregressive Task**:
  ```bash
  python Non-Autoregressive-Task.py
  ```
- **Autoregressive Task using CodeT5**:
  ```bash
  python Autoregressive-T5.py
  ```
- **Autoregressive Task using CodeGPT**:
  ```bash
  python Autoregressive-GPT.py
  ```

These scripts evaluate the models using the extended CodeBLEU score implemented in `codebleu/codebleu.py`.

### 2. Impact of Quantization on Memorization
To reproduce the results from Chapter 5, quantization can be enabled in the scripts by passing the `--quantization` flag:

- **Non-Autoregressive Task** with quantization:
  ```bash
  python Non-Autoregressive-Task.py --quantization True
  ```
- **Autoregressive Task using CodeT5** with quantization:
  ```bash
  python Autoregressive-T5.py --quantization True
  ```
- **Autoregressive Task using CodeGPT** with quantization:
  ```bash
  python Autoregressive-GPT.py --quantization True
  ```

### 3. Few-Shot Learning and Prompt Tuning for Memorization Analysis
To reproduce the results from Chapter 6, navigate to the `Few-shot` and `Prompt-Tuning` directories and run the corresponding scripts:

- **Few-Shot Learning**:
  ```bash
  cd Few-shot
  python fsl-codeT5.py
  python fsl-codeGPT.py
  ```

- **Prompt Tuning**:
  ```bash
  cd Prompt-Tuning
  python codeT5-PT.py
  python codeGPT-PT.py
  python codeBERT-PT.py
  ```


## Extended CodeBLEU Score

The implementation of the extended CodeBLEU score can be found at:
```bash
./PROJECT/Evaluating Memorization Across LLM Architectures/codebleu/codebleu.py
```

## Dependencies

Please ensure you have the following main dependencies installed:

You can install the required packages using:
``` bash
pip install -r requirements.txt
```
## Arguments

All the scripts accept various arguments to customize the experiments. Here are the common arguments:

```python
parser.add_argument('--prog_lang', type=str, default='python', choices=['python', 'java', 'javascript', 'ruby'], help='Programming language of the code snippets')
parser.add_argument('--mask_ratio', type=float, default=0.75, help='Masking ratio for the code snippets')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory for the extracted content')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the extraction process on')
parser.add_argument('--autoregressive_model_name', type=str, default='Salesforce/codet5-large', help='Name of the autoregressive model to use for extracting content')
parser.add_argument('--non_autoregressive_model_name', type=str, default='microsoft/codebert-base-mlm', help='Name of the non-autoregressive model to use for extracting content')
parser.add_argument('--quantization', type=bool, default=False, help='Whether to quantize the model or not')
```

For few-shot learning:

```python
parser.add_argument('--number_of_shot', type=int, default=0, help='Number of shots to use for the extraction process')
parser.add_argument('--number_of_examples', type=int, default=10, help='Number of examples to try few-shot learning on')
```

## Conclusion

This repository provides the necessary scripts and instructions to reproduce the experiments discussed in the thesis "Investigating Memorization in Code-Based Large Language Models." Please ensure that the appropriate dependencies are installed, and the dataset is correctly set up as described before running the experiments.

For further inquiries, please refer to the thesis document or contact the author.

## Notes

- Large datasets and model checkpoints are not included in this archive due to size constraints.
- For reproducibility, please use the same versions of the libraries and models as specified in the thesis.
- Some experiments may require significant computational resources. Consider using GPU acceleration when available.

## Author

Vaikunth Guruswamy
