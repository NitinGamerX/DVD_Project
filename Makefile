# Regenerates every deliverable from the model file, in order (deliverable 9).
# `make all` on a fresh clone must reproduce data, models, tables, plots and report.
PY := .venv/bin/python
SCRIPTS := scripts

# step target -> script
STEPS := \
  check:00_check_env \
  fet-data:01_generate_fet_data \
  data-check:02_data_check \
  train:03_train_models \
  gates-spice:04_gate_spice \
  gates-ml:05_gate_ml \
  cla-spice:06_cla_spice \
  cla-ml:07_cla_ml \
  analysis:08_error_analysis

.PHONY: help setup all report clean-scratch $(foreach s,$(STEPS),$(firstword $(subst :, ,$(s))))

help:
	@echo "make setup        one-time environment setup (./setup.sh)"
	@echo "make check        Phase 0  environment + config check"
	@echo "make fet-data     Phase 1  four single-FET datasets (deliverables 1, 2)"
	@echo "make data-check   Phase 2  Kirchhoff check + trend plots (deliverable 3)"
	@echo "make train        Phase 3  model comparison + chosen models (deliverables 4, 5)"
	@echo "make gates-spice  Phase 4  gate SPICE reference, 18 cases x 200 corners"
	@echo "make gates-ml     Phase 5  gate ML prediction + table + parity (deliverables 6, 7)"
	@echo "make cla-spice    Phase 6  3-bit CLA SPICE reference"
	@echo "make cla-ml       Phase 6  3-bit CLA ML prediction + metrics (deliverable 8)"
	@echo "make analysis     Phase 7  error analysis for the report"
	@echo "make report       build report/main.pdf"
	@echo "make all          everything above, in order"

setup:
	./setup.sh

define STEP_RULE
$(1):
	@if [ -f $(SCRIPTS)/$(2).py ]; then $(PY) $(SCRIPTS)/$(2).py; \
	 else echo "$(SCRIPTS)/$(2).py not written yet"; exit 1; fi
endef
$(foreach s,$(STEPS),$(eval $(call STEP_RULE,$(firstword $(subst :, ,$(s))),$(lastword $(subst :, ,$(s))))))

all: check fet-data data-check train gates-spice gates-ml cla-spice cla-ml analysis report

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-scratch:
	rm -rf work/ report/*.aux report/*.log report/*.out report/*.toc report/*.fls report/*.fdb_latexmk
