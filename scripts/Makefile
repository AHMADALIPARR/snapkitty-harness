# SPDX-License-Identifier: LicenseRef-SnapKitty-Commercial-1.0
# Copyright (C) 2026 SnapKitty Collective
#
# Makefile — snapkitty-harness reproduction pipeline.
#
#   make check     syntax-check scripts + run unit tests
#   make train     LoRA SFT on data/train.jsonl + data/redteam.jsonl
#   make merge     merge adapters into the base model
#   make convert   convert merged checkpoint to f16 GGUF (needs llama.cpp)
#   make quantize  quantize f16 GGUF to Q4_K_M (needs llama.cpp)
#   make verify    verify the final GGUF header
#   make all       train -> merge -> convert -> quantize -> verify
#
# Knobs (override on the command line):
#   DATA_DIR, BASE_MODEL, ADAPTER, MERGED, LLAMACPP, F16GGUF, OUTGGUF, GGUF

DATA_DIR   ?= ../data
BASE_MODEL ?= nvidia/Nemotron-Mini-4B-Instruct
ADAPTER    ?= ../out/adapter
MERGED     ?= ../out/merged
OUTGGUF    ?= ../out/gguf
F16GGUF    ?= $(OUTGGUF)/snapkitty-harness-f16.gguf
GGUF       ?= $(OUTGGUF)/snapkitty-harness.Q4_K_M.gguf
LLAMACPP   ?=

.PHONY: check train merge convert quantize verify all clean

check:
	python3 -m py_compile train.py verify.py ../tests/test_verify.py
	bash -n merge_and_convert.sh
	python3 -m pytest ../tests/ -q

train:
	python3 train.py --data-dir "$(DATA_DIR)" \
		--base-model "$(BASE_MODEL)" --out "$(ADAPTER)"

merge:
	ADAPTER="$(ADAPTER)" BASE_MODEL="$(BASE_MODEL)" MERGED="$(MERGED)" \
		./merge_and_convert.sh merge

convert:
	LLAMACPP="$(LLAMACPP)" MERGED="$(MERGED)" OUTGGUF="$(OUTGGUF)" \
		./merge_and_convert.sh convert

quantize:
	LLAMACPP="$(LLAMACPP)" F16GGUF="$(F16GGUF)" OUTGGUF="$(OUTGGUF)" \
		./merge_and_convert.sh quantize

verify:
	python3 verify.py "$(GGUF)"

all: train merge convert quantize verify

clean:
	rm -rf ../out
