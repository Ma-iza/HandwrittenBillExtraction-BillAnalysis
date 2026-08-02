# Handwritten Receipt Information Extraction using Multimodal LLMs

## Overview

This project evaluates the performance of multimodal Large Language Models (LLMs) for extracting structured information from handwritten Indian receipts.

While LLMs perform well on digital invoices, handwritten receipts introduce challenges such as variations in handwriting, receipt formats, image quality, and missing information. This project compares different vision-language models based on extraction accuracy and API cost.



## Models Evaluated

The following multimodal models were evaluated:

| Model                       | Provider   |
| --------------------------- | ---------- |
| NVIDIA Nemotron Nano 12B VL | OpenRouter |
| Command-A-Vision-07-2025    | Cohere     |
| Qwen3.6-27B                 | Groq       |

---

## Dataset

A custom dataset of **12 handwritten Indian receipts** was created.

---

## Evaluation Framework

A custom evaluation framework was developed to compare model-generated outputs against ground truth values.

The framework evaluates:

* Field-level accuracy
* Model-wise performance
* API usage cost

---

## Running the Evaluation

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the evaluation script.

```bash
python eval.py
```

The script generates field-wise accuracy results for each evaluated model.

---

## Key Findings

* Qwen3.6-27B provided the best balance between accuracy and cost.
* Command-A-Vision achieved strong performance on several fields but had significantly higher API costs.
* Nemotron Nano 12B VL provided competitive accuracy with moderate cost.

---

## Limitations

* The evaluation dataset contains only 12 receipts.
* Results may vary on larger and more diverse receipt datasets.
* Cost calculations are based on reported token usage and provider pricing.

---

## Conclusion

This project demonstrates the capability of multimodal LLMs for handwritten receipt information extraction and provides a framework to compare models based on both accuracy and cost efficiency.
