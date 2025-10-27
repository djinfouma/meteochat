# MeteoChat: semiautomatic generation of environmental accessible reports  via LLM-based chatbot

## Overview

**MeteoChat** is an AI-powered system designed to automatically generate environmental reports from meteorological data.
It leverages Large Language Models (LLMs) optimized through Fine-tuning and Retrieval-Augmented Generation (RAG) to produce accurate, accessible, and human-readable environmental documents. The project aims to assist both domain experts and non-technical users by transforming raw environmental datasets into meaningful insights and narrative reports.

## Architecture

The system operates in two stages: in the first, an environmental expert defines questions and answers related to different metrics, such as temperature and precipitation, which are used to fine-tune a GPT-4o model specialized in environmental data. In the second stage, the optimized model is integrated into a chatbot that retrieves relevant data using the RAG technique and generates accurate and context-aware responses. Finally, it produces complete .docx reports with summaries, graphs (Plotly), and conclusions tailored to expert or general users.

## User's Guide

To use **MeteoChat**, you first need to install **Python** (version **3.10** or **3.11** is recommended) and several additional libraries:  
`Chroma`, `Docx`, `Flask`, `LangChain`, `OpenAI`, `Pandas`, `Plotly`, and `Scipy`.  

These can be installed via the terminal using:

```bash
pip install library_name
```

To ensure correct execution and to manage dependencies independently, MeteoChat is developed and executed within a **Python virtual environment**.  

You can create and activate it using the following commands:

```bash
python -m venv your_env_name
# On macOS/Linux:
source your_env_name/bin/activate
# On Windows:
your_env_name\Scripts\activate
```

Using a virtual environment is not mandatory but **strongly recommended** to maintain an isolated and controlled workspace.

After the environment setup, select the raw data you wish to analyze and clean them by running:

```bash
python data_cleaning.py
```

Some parameters in the script must be adjusted according to the metric you intend to process.

Once the data are cleaned, you can perform statistical calculations using:

```bash
python statistics.py
```

This script should also be customized based on the selected metric.

To perform the **fine-tuning**, you need valid **OpenAI credentials** (to be stored in the `secrets.json` file) and a JSONL training file (for example, `general-public_expert.jsonl`).  

After preparing these files, run:

```bash
python fine-tune-model-azure_expert.py
```

This script customizes the LLM using the set of questions, procedures, and answers defined in the JSONL file.

Next, edit the chatbot script by specifying the data folder, for example:

```python
loader_prec = DirectoryLoader('output/precipitazioni/')
```

and define the fine-tuned model name in the `AzureChatOpenAI` class:

```python
model="fine_tuned_model_name"
```

Finally, launch MeteoChat by running:

```bash
python app.py
```

In the terminal, a message similar to the following will appear:

```
* Running on http://chatbot_address
```

Open this address in your browser to start using **MeteoChat** as an interactive conversational agent.

## Contributing

MeteoChat is an open-source initiative that encourages community collaboration. Whether you’re enhancing features, fixing bugs, or refining documentation, your contributions are highly appreciated.

## License

MeteoChat is released under the MIT License, allowing both personal and commercial use with minimal restrictions. See the LICENSE file in the repository for the complete license text.

## Authors

MeteoChat was implemented by **Alessandra Scariot** as a master's thesis project in Digital Humanities, under the supervision of professor **Angelica Lo Duca**.
