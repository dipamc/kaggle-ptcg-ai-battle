# Native PTCG backend — CPU-side builds (env tests + parity). The full
# CUDA _C extension builds on a CUDA host via build_native.sh.
#
# macOS (arm64):  make -C native test    (links data/cg/libcg.dylib)
# Linux:           make -C native test    (links data/cg/libcg.so)

UNAME := $(shell uname -s)
ROOT := ..
CG_DIR := $(ROOT)/data/cg
ifeq ($(UNAME),Darwin)
  CG_LIB := $(CG_DIR)/libcg.dylib
  # NOTE: this -rpath is inert. libcg.dylib's install name is a bare
  # "libcg.dylib" (otool -D), not "@rpath/libcg.dylib", and dyld only consults
  # rpath for @rpath-prefixed names. Hence RUNENV below: the run targets need
  # DYLD_LIBRARY_PATH, which is also why docs/deck-pool.md sets it.
  RPATH := -Wl,-rpath,@executable_path/$(CG_DIR)
  RUNENV := DYLD_LIBRARY_PATH=$(abspath $(CG_DIR))
else
  CG_LIB := -L$(CG_DIR) -lcg -Wl,-rpath,$(abspath $(CG_DIR))
  RPATH :=
  RUNENV :=
endif

CC ?= cc
CFLAGS := -O2 -g -std=c11 -Wall -Wextra -Wno-unused-parameter \
          -Ivendor -Iptcg -pthread
SRC := ptcg/obs.c ptcg/tables.c ptcg/tracker.c ptcg/encoder.c ptcg/env.c \
       vendor/cJSON.c

build/test_env: $(SRC) ptcg/test_env.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/test_env.c $(CG_LIB) $(RPATH) -lm -o $@

build/parity_replay: $(SRC) ptcg/parity_replay.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/parity_replay.c $(CG_LIB) $(RPATH) -lm -o $@

build/test_reload: $(SRC) ptcg/test_reload.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/test_reload.c $(CG_LIB) $(RPATH) -lm -o $@

build/test_deck_lock: $(SRC) ptcg/test_deck_lock.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/test_deck_lock.c $(CG_LIB) $(RPATH) -lm -o $@

build/test_deck_weights: $(SRC) ptcg/test_deck_weights.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/test_deck_weights.c $(CG_LIB) $(RPATH) -lm -o $@

build/test_stall: $(SRC) ptcg/test_stall.c ptcg/ptcg_env.h | build
	$(CC) $(CFLAGS) $(SRC) ptcg/test_stall.c $(CG_LIB) $(RPATH) -lm -o $@

build:
	mkdir -p build

test: build/test_env
	cd $(ROOT) && $(RUNENV) PTCG_TABLES=native/ptcg_tables.bin ./native/build/test_env 16 20000

parity: build/parity_replay
	cd $(ROOT) && $(RUNENV) PTCG_TABLES=native/ptcg_tables.bin \
	  ./native/build/parity_replay native/parity_stream.jsonl native/parity_rows.bin

# Deck-append contract test. Build the three blobs first -- the exact
# export_tables.py invocations are in docs/deck-pool.md.
#   make -C native reload V1=<v1.bin> GROWN=<v1+3.bin> INSERTED=<reorder.bin>
reload: build/test_reload
	cd $(ROOT) && $(RUNENV) PTCG_TABLES=$(V1) ./native/build/test_reload $(V1) $(GROWN) $(INSERTED)

# Does a swapper holding the write lock actually stall the env threads?
#   make -C native decklock V1=<v1.bin> GROWN=<v1+3.bin>
decklock: build/test_deck_lock
	cd $(ROOT) && $(RUNENV) PTCG_TABLES=$(V1) ./native/build/test_deck_lock $(V1) $(GROWN)

# Does per-deck weighting actually shape the draw, and fail safe?
#   make -C native deckweights V1=<blob> GROWN=<grown blob>   (GROWN optional)
deckweights: build/test_deck_weights
	cd $(ROOT) && $(RUNENV) PTCG_TABLES=$(V1) ./native/build/test_deck_weights $(V1) $(GROWN)

clean:
	rm -rf build

.PHONY: test parity reload decklock deckweights clean
