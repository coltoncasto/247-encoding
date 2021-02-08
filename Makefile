# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
FILE_ERP := tfserp_main2
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

# 625 Electrode IDs
E_LIST := $(shell seq 1 55)
# E_LIST := 11 17 20

# 676 Electrode IDs
# E_LIST := $(shell seq 1 1)
# E_LIST := $(shell seq 1 63)

# Choose the subject to run for
SID := 625
# SID := 676

NPERM := 1

# Choose the lags to run for.
LAGS := {-5000..5000..25}
# LAGS := 0

# Choose which set of embeddings to use
EMB := glove50
# EMB := gpt2-xl

# Choose the window size to average for each point
WS := 200
WS_ERP := 4
CNXT_LEN := 1024

# Choose to align the vocab with another set of embeddings
ALIGN_WITH := gpt2-xl
ALIGN_TGT_CNXT_LEN := 1024

# Specify the minimum word frequency
MWF := 1

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to PCA the embeddings before regressing or not
# PCA := --pca-flag
PCA_TO := 50

# Choose how to split the datum, which subset to look at
SPLIT := --split-flag
SPLIT_BY := correct
SPLIT_BY := incorrect

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := echo
CMD := sbatch submit1.sh
CMD := python


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
	$(CMD) code/$(FILE).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
		--window-size $(WS) \
		--word-value $(WV) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		$(PCA) \
		--reduce-to $(PCA_TO) \
		$(SH) \
		$(PSH) \
		$(SPLIT) \
		--split-by $(SPLIT_BY) \
		--output-prefix $(DT)-$(USR)-$(WS)ms-$(WV)-$(SPLIT_BY); \


# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--sid $(SID) \
			--electrodes $$elec \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			$(SPLIT) \
			--split-by $(SPLIT_BY) \
			--output-prefix $(DT)-$(USR)-$(WS)ms-$(WV)-$(SPLIT_BY); \
	done

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM);

# -----------------------------------------------------------------------------
# Plotting encoding
# -----------------------------------------------------------------------------

plot-encoding1:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--sid $(SID) \
			--electrodes $(E_LIST) \
			--input-directory \
				20210208-1451-ca-200ms-all-correct-625-glove50-cnxt-1024-pca_0d \
				20210208-1449-ca-200ms-all-incorrect-625-glove50-cnxt-1024-pca_0d \
			--labels \
				enc-correct \
				enc-incorrect \
			--output-file-name \
				'$(DT)-$(SID)-correct-incorrect-enc'

# -----------------------------------------------------------------------------
# ERP
# -----------------------------------------------------------------------------

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-erp:
	mkdir -p logs
	$(CMD) code/$(FILE_ERP).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
		--window-size $(WS_ERP) \
		--word-value $(WV) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		$(PCA) \
		--reduce-to $(PCA_TO) \
		$(SH) \
		$(PSH) \
		$(SPLIT) \
		--split-by $(SPLIT_BY) \
		--output-prefix $(DT)-$(USR)-$(WS_ERP)ms-$(WV)-$(SPLIT_BY); \


# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-erp-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE_ERP).py \
			--sid $(SID) \
			--electrodes $$elec \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS_ERP) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--output-prefix $(DT)-$(USR)-$(WS_ERP)ms-$(WV)-$(SPLIT_BY); \
	done

pca-on-embedding-erp:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM);

# -----------------------------------------------------------------------------
# Plotting ERP
# -----------------------------------------------------------------------------

plot-erp:
	mkdir -p results/erp/figures
	python code/tfserp_plots.py \
			--sid $(SID) \
			--electrodes $(E_LIST) \
			--input-directory \
				20210208-1451-ca-200ms-all-correct-625-glove50-cnxt-1024-pca_0d \
				20210208-1449-ca-200ms-all-incorrect-625-glove50-cnxt-1024-pca_0d \
			--labels \
				erp-correct \
				erp-incorrect \
			--output-file-name \
				'$(DT)-$(SID)-erp-correct-incorrect'


# -----------------------------------------------------------------------------
#  Misc. targets
# -----------------------------------------------------------------------------

# If you have pickled the data yourself, then you can just link to it
link-data:
	ln -fs $(shell dirname `pwd`)/247-pickling/results/* data/

sync-plots:
	rsync -aPv --delete \
	    results/figures/ \
	    /tigress/$(USER)/247-encoding-results/figures

sync-results:
	rsync -aP --delete \
	    results/ \
	    /tigress/$(USER)/247-encoding-results
