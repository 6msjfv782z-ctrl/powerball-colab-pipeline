# Powerball Colab Pipeline

This updated repository now includes:
- sample_history.csv: a tiny example CSV you can use to test the notebook if scraping fails.
- colab_pipeline.py: extended script with heavier-search parameters, GA/annealing functions, and hosted Hugging Face inference scoring.
- powerball_colab.ipynb: notebook that runs the script in Colab.

Heavier-search tuning (environment variables)
- MONTE_CARLO_POOL: number of Monte Carlo candidates (default 20000)
- GA_POPULATION, GA_GENERATIONS: GA tuning (if DEAP installed)
- ANNEAL_ITER: annealer iterations
- REPEAT_RUNS: how many full pipeline runs to repeat and reconcile

Hosted LLM (Hugging Face)
- To enable hosted-LLM steps, set HUGGINGFACE_API_TOKEN in the Colab environment. The notebook will attempt to use HF_MODELS listed in the script.
- Hosted-model scoring is batched; monitor your HF quotas.

Open in Colab link:
https://colab.research.google.com/github/6msjfv782z-ctrl/powerball-colab-pipeline/blob/main/powerball_colab.ipynb
