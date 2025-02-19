# Agent-RAG: High-Quality Retrieval-Augmented Generation System

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/yourusername/agent-rag/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/agent-rag?style=social)](https://github.com/yourusername/agent-rag/stargazers)

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview

Agent-RAG is a high-quality, agent-based Retrieval-Augmented Generation (RAG) system designed for efficient and accurate document retrieval and question answering. The system leverages advanced techniques such as hierarchical node parsing, sentence window indexing, and auto-merging retrieval to provide precise and context-aware responses. With its modular architecture and flexible configuration, Agent-RAG is suitable for various applications, including academic research, enterprise knowledge management, and automated customer support.

Key Features:
- **Accurate Question Answering**: Utilizes advanced indexing and retrieval techniques for precise responses.
- **Flexible Indexing**: Supports multiple indexing strategies including basic fixed-size, auto-merging, and sentence window indexing.
- **Efficient Retrieval**: Implements hierarchical node parsing and sentence window retrieval for optimized performance.
- **Easy Deployment**: Provides a simple API interface for integration with existing systems.

## Requirements

- Python 3.8+
- GPU with at least 16GB VRAM (recommended)
- Weaviate (for vector storage)

## Installation

1. Clone the repository:

git clone https://github.com/dalong0514/easy-rag.git

cd easy-rag

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up Weaviate:

```
docker compose up -d
```

## Usage

### 1. API

Start the FastAPI server:

cd easy-rag

```
python -m api.main
```

### 2. Building Indexes

You can build different types of indexes using the API:

```
curl -X POST "http://127.0.0.1:8001/build-index" \
-H "Content-Type: application/json" \
-d '{
    "input_path": "/Users/Daglas/dalong.github/dalong.readnotes/20250101复制书籍/2025002The-Art-of-Doing-Science-and-Engineering",
    "index_name": "Book2025002The_Art_of_Doing_Science_and_Engineering",
    "index_type": "basic",
    "file_extension": "md"
}'
```

## Project Structure

agent-rag/
├── api/ # FastAPI application
│ └── main.py # API endpoints
├── src/ # Core functionality
│ ├── indexing.py # Index building functions
│ ├── retrieval.py # Document retrieval functions
│ └── utils.py # Utility functions
├── eval/ # Evaluation scripts
│ └── utils_eval.py # Evaluation utilities
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Citation

If you use Agent-RAG in your research, please cite it as:

bibtex
@software{EasyRAG,
author = {Feng Dalong},
title = {Agent-RAG: High-Quality Agent-Based Retrieval-Augmented Generation System},
year = {2024},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/dalong0514/easy-rag}}
}


## Acknowledgement

We would like to thank the open-source community for their contributions to the libraries and tools that made this project possible, including LlamaIndex, Weaviate, and FastAPI.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=dalong0514/easy-rag&type=Date)