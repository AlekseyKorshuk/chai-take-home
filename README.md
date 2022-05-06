# Chai Research’s take-home | Report

## Working pipeline

- [x] Read about Meta's OPT
- [x] Analyze weights structure
- [x] Fork Transformers library and adapt it to the new model
- [x] Convert default weights to new transformer's architecture
- [x] Create config.json for new model
- [x] Upload model and tokenizer to Hugging Face Hub
- [x] Create FastAPI integration for model inference
- [x] Upload image to Docker Hub
- [x] Prepare Kubernetes YAMLs
- [x] Upload to the server and test inference
- [x] Write README.md
- [x] Write how to reduce cost with Optimum Transformers

## How to use

Before all it is important to install transformers with this line:

```bash
pip install git+https://github.com/AlekseyKorshuk/transformers.git@opt
```

### Convert model by your own and use it

With Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlekseyKorshuk/chai-take-home/blob/main/notebooks/opt_integration.ipynb)

### With pipeline

```python
from transformers import pipeline

pipe = pipeline("text-generation", "AlekseyKorshuk/opt-350m")
results = pipe("Today is a beautiful day and I want to")
```

### With Docker

```bash
docker pull alekseykorshuk/chai-take-home
docker run alekseykorshuk/chai-take-home:main 
```

### With Kubernetes

```bash
kubectl apply -f api.yaml
```

## How to reduce costs

The main thing is to accelerate model inference, so reduce the amount of servers to archive the same bandwidth.

The process is in 3 steps:

* Convert Pytorch model to a graph
* Optimize the graph
* Deploy the graph on a performant inference server

I made a great library that combines all these steps together:
[Optimum Transformers](https://github.com/AlekseyKorshuk/optimum-transformers).

```python
from optimum_transformers import pipeline

nlp = pipeline("sentiment-analysis", use_onnx=True, optimize=True)
nlp("Chai is the best startup ever!")
# [{'label': 'POSITIVE', 'score': 0.999721109867096}]  
```

Here is a quick overview how it was done and a lot of details on how each step can be performed:
[link](https://medium.com/@alekseykorshuk/optimum-transformers-61d4c61e5754).

To use the new OPT architecture as easy as possible, will need to do the following:

* Add OPT architecture to transformers officially
* Add ONNX graph converter with new OPT architecture to Transformers & ONNX
* Add new converter to Optimum

Everything else has already been done by me in [Optimum Transformers](https://github.com/AlekseyKorshuk/optimum-transformers). .