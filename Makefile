# Auto-discover compilable examples (those defining create_network_parts)
COMPILABLE := $(shell grep -rl 'def create_network_parts' examples/*.py \
    | sed 's|examples/||; s|\.py||')
HF_BUNDLES := $(addsuffix _hf_bundle/model.safetensors.index.json, $(COMPILABLE))

.PHONY: compile
compile: $(HF_BUNDLES)

%_hf_bundle/model.safetensors.index.json: examples/%.py examples/compile.py torchwright/compiler/*.py torchwright/graph/*.py
	uv run python -m examples.compile $* --no-demo

.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy .

# Guard the standalone lock the Modal --frozen build installs from.  It cannot
# be refreshed by uv inside the workspace, so it drifts silently; this check
# (a prerequisite of `make test`) fails fast if a Modal-synced group gained a
# dependency the lock is missing.  Fix drift with `make modal-lock`.
.PHONY: check-modal-lock
check-modal-lock:
	@uv run python -m scripts.check_modal_lock

# Regenerate torchwright/uv.lock OUT OF the workspace (uv would otherwise resolve
# to the umbrella root).  Mirrors the only supported way to refresh this lock.
.PHONY: modal-lock
modal-lock:
	@echo "Regenerating standalone torchwright/uv.lock out-of-workspace…"
	@TMP="$$(mktemp -d)" ; \
		cp pyproject.toml uv.lock "$$TMP/" ; \
		( cd "$$TMP" && uv lock ) ; \
		cp "$$TMP/uv.lock" uv.lock ; \
		rm -rf "$$TMP" ; \
		echo "Updated torchwright/uv.lock — review the diff and commit."

.PHONY: test
test: check-modal-lock
	@bash -c ' \
		LOGFILE=/tmp/torchwright-test-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright-test.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running tests on Modal ===" | tee -a "$$LOGFILE" ; \
		echo "=== Monitor: make test-logs ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		uv run modal run modal_test.py \
			--file $(if $(FILE),$(FILE),tests) \
			$(if $(ARGS),--args "$(ARGS)") \
			2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Tests finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'

.PHONY: test-logs
test-logs:
	@tail -f /tmp/torchwright-test.log

# CI entry point (.github/workflows/ci.yml): full suite on the runner's CPU,
# synced from the standalone lock like the Modal image.  Guarded to CI because
# inside the umbrella workspace `uv sync` operates on the shared venv instead
# (dropping opt-in packages like torchwright_doom's onnxruntime-gpu), and a
# full local suite run belongs on Modal anyway (see test-local's guard).
.PHONY: test-ci
test-ci:
	@if [ -z "$$CI" ]; then \
		echo "Error: test-ci is the CI entry point (runs the full suite locally)." >&2 ; \
		echo "       Use 'make test' (Modal) or 'make test-local FILE=...' instead." >&2 ; \
		exit 2 ; \
	fi
	uv sync --group test-onnx
	uv run --no-sync pytest tests --device cpu

.PHONY: test-local
test-local:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE=<path> is required for test-local." >&2 ; \
		echo "       test-local runs pytest on the local machine and must target" >&2 ; \
		echo "       a single file to avoid accidentally running the whole suite" >&2 ; \
		echo "       (which belongs on Modal via 'make test')." >&2 ; \
		echo "Example: make test-local FILE=tests/graph/test_embedding.py" >&2 ; \
		exit 2 ; \
	fi
	uv run pytest $(FILE) $(ARGS)

.PHONY: measure-noise
measure-noise:
	uv run python -m scripts.measure_op_noise $(ARGS)

.PHONY: modal-run
modal-run:
	@if [ -z "$(MODULE)$(SCRIPT)" ]; then \
	    echo "Error: MODULE=<dotted.name> or SCRIPT=<path> required." >&2 ; \
	    echo "Example: make modal-run MODULE=scripts.investigate_phase_e" >&2 ; \
	    exit 2 ; \
	fi
	@bash -c ' \
		LOGFILE=/tmp/torchwright-modal-run-$$(date +%Y%m%d-%H%M%S).log ; \
		ln -sfn "$$LOGFILE" /tmp/torchwright-modal-run.log ; \
		echo "=== Log file: $$LOGFILE ===" | tee "$$LOGFILE" ; \
		echo "=== Running on Modal ===" | tee -a "$$LOGFILE" ; \
		start=$$(date +%s) ; \
		uv run modal run modal_run.py \
		    $(if $(MODULE),--module $(MODULE)) \
		    $(if $(SCRIPT),--script $(SCRIPT)) \
		    $(if $(ARGS),--args "$(ARGS)") \
		    $(if $(CPU_ONLY),--cpu-only) \
		    2>&1 | tee -a "$$LOGFILE" ; \
		rc=$${PIPESTATUS[0]} ; \
		end=$$(date +%s) ; \
		echo "" | tee -a "$$LOGFILE" ; \
		echo "=== Finished in $$((end - start))s (exit $$rc) ===" | tee -a "$$LOGFILE" ; \
		echo "=== Log file: $$LOGFILE ===" | tee -a "$$LOGFILE" ; \
		exit $$rc \
	'
