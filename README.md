# Powerball Colab Pipeline

This repository contains a Google Colab-ready pipeline to fetch Oz Lotteries Powerball draw history, compute statistical features, run multiple independent generators (frequency, overdue, pair/triple, Monte Carlo), integrate optional hosted open-source LLM critic/generator via Hugging Face Inference (Mistral, Llama 2), run a heavier search (GA / simulated annealing), and reconcile two independent runs into a validated 8-game ticket output (JSON + human-readable). The notebook is packaged to run on Colab using hosted inference (Hugging Face) when you provide an API token.

Files in this repo:
- colab_pipeline.py    -- the main runnable pipeline script
- powerball_colab.ipynb -- lightweight Colab notebook that downloads and runs the script
- requirements.txt    -- Python dependencies
- README.md            -- this file

Quick start (Open-in-Colab link):
- Open this notebook in Colab: https://colab.research.google.com/github/6msjfv782z-ctrl/powerball-colab-pipeline/blob/main/powerball_colab.ipynb

How to run in Colab (hosted inference option):
1. (Optional but recommended) Create a free Hugging Face account and copy an API token: https://huggingface.co/settings/tokens
2. In Colab, set the token as an environment variable before running the notebook cells, e.g.:
   import os
   os.environ['HUGGINGFACE_API_TOKEN'] = 'hf_xxx'
   Or in the notebook UI, use Colab "Edit > Notebook settings" or the token input cell.
3. Run the single notebook cell. The pipeline will:
   - attempt to scrape ozlotteries for Powerball history; if scraping fails, it prompts for CSV upload
   - build features and candidate pools
   - run CPU/GPU intensive GA/annealing steps if configured (longer runtime)
   - if HUGGINGFACE_API_TOKEN is provided, call hosted models sequentially (Mistral/Llama variants) to generate/critique candidates
   - run two independent passes and reconcile final 8 lines
4. The notebook prints a JSON object first and then exactly 8 human-readable ticket lines.

Notes and warnings
- This tool automates statistical analysis and model-based candidate generation; it does not guarantee winning the jackpot. The combinatorial odds are large.
- Hosted LLM inference may be subject to model licensing and usage limits; Mistral and Llama 2 model availability on Hugging Face may vary.

If you want, I can also add a sample official CSV for testing, or extend the GA search parameters—tell me which you prefer.
