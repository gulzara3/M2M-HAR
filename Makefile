.PHONY: install test analysis figures all clean

install:
	pip install -e ".[analysis,dev]"

test:
	pytest -q

analysis:  ## regenerate statistics, tables and data figures from the raw notebook outputs
	python analysis/make_paper_figures.py results/notebook_outputs results/paper_analysis
	python analysis/make_extra_figures.py results/notebook_outputs results/paper_analysis results/paper_analysis/figs

figures: analysis  ## collect the numbered paper figures into figures/
	python analysis/collect_figures.py

diagrams:  ## re-render Fig. 1 and Fig. 2 from HTML (needs Node.js + Playwright with Chromium)
	cd analysis/diagrams && node render.js $$PWD/fig_framework.html $$PWD/fig_framework.png $$PWD/fig_framework.pdf
	cd analysis/diagrams && node render.js $$PWD/fig_encoder.html $$PWD/fig_encoder.png $$PWD/fig_encoder.pdf

all: test figures

clean:
	rm -rf .pytest_cache src/*.egg-info **/__pycache__
