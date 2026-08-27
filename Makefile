# HEAAF -- reproduction targets.
#
# Every result in the paper comes from scripts/stage.py.  There is deliberately
# no second, monolithic runner: two code paths to the same table is how a
# manuscript ends up quoting numbers no one can regenerate.

SEEDS      ?= 0,1,2,3,4,5,6,7
ABL_SEEDS  ?= 0 1 2 3 4
DRIFT_SEEDS?= 0 1 2
EVENTS     ?= 140000
EPISODES   ?= 90000
BUDGET     ?= 0.01

.PHONY: all quick full data train main explain ablation drift figures tables test clean help

help:
	@echo "make quick    ~3 min smoke run at reduced scale"
	@echo "make full     ~90 min full evaluation reported in the paper"
	@echo "make figures  regenerate figures from persisted results"
	@echo "make tables   regenerate LaTeX tables + numbers.json"
	@echo "make test     run the invariant test suite"

all: full figures tables

quick:
	python scripts/stage.py data  --events 15000
	python scripts/stage.py train --seed 0 --episodes 8000
	python scripts/stage.py main  --seeds 0 --budget $(BUDGET) --pareto 0
	python scripts/stage.py explain      --seed 0 --n-instances 40
	python scripts/stage.py groupexplain --seed 0 --n-instances 60
	python scripts/stage.py latency      --seed 0
	python scripts/make_figures.py
	python scripts/make_tables.py

full: data train main explain ablation drift figures tables

data:
	python scripts/stage.py data --events $(EVENTS)

train:
	@for s in $$(echo $(SEEDS) | tr ',' ' '); do \
	  python scripts/stage.py train --seed $$s --events $(EVENTS) --episodes $(EPISODES); \
	done

main:
	python scripts/stage.py main --seeds $(SEEDS) --budget $(BUDGET) --pareto 3

explain:
	python scripts/stage.py explain      --seed 0 --n-instances 300
	python scripts/stage.py groupexplain --seed 0 --n-instances 500
	python scripts/stage.py latency      --seed 0

ablation:
	@for s in $(ABL_SEEDS); do \
	  python scripts/stage.py ablation --which -1 --seed $$s \
	    --events $(EVENTS) --episodes $(EPISODES) --budget $(BUDGET); \
	done
	python scripts/stage.py tables

drift:
	@for s in $(DRIFT_SEEDS); do \
	  python scripts/stage.py drift --seed $$s --events $(EVENTS) --episodes $(EPISODES); \
	done

figures:
	python scripts/make_figures.py

tables:
	python scripts/make_tables.py

test:
	pytest -q

clean:
	rm -rf results/logs/* results/tables/* results/figures/* data/processed/*
	@touch results/logs/.gitkeep results/tables/.gitkeep \
	       results/figures/.gitkeep data/processed/.gitkeep
