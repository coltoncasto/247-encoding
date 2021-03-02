# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
FILE_ERP := tfserp_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

# 625 Electrode IDs
E_LIST := $(shell seq 1 140)
# E_LIST := 11 12 13 14 17 20 21 27 29 30 31 32 33 35 36 37 38 40 42 43 44 45 46 47 49 50 51 52 
# E_LIST := 11 19 24 33 40 46 47 48 49 55

# 676 Electrode IDs
# E_LIST := $(shell seq 1 1)
# E_LIST := $(shell seq 1 63)

# Choose the subject to run for
SID := 625
SID := 676

NPERM := 1
# NPERM := 10

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
SPLIT_BY := all
SPLIT_BY := correct
SPLIT_BY := incorrect
# SPLIT_BY := incorrect-w-predicted

# Choose significant electrodes based on threshold
SIG := --sig-flag
SIG_THRESH := .12 

# prod/comp flags for plotting
PROD := --prod
COMP := --comp

# combine subjects for plotting 
# e.g., take four directors and combine 1/3 and 2/4
CMB := --combine-subjects

# only look at one conversation for significant encodings
CID := --conversation-id-flag
CID_NUM := 63

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
		--sig-elec-file \
			20210302-0855-676-glove50-sig-elecs-1-convo_comp.csv \
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
		$(CID) \
		--conversation-id $(CID_NUM) \
		--output-prefix $(DT)-$(USR)-$(WS)ms-$(SPLIT_BY)-test; \


# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--sid $(SID) \
			--electrodes $$elec \
			--sig-elec-file \
				20210302-0855-676-glove50-sig-elecs-1-convo_comp.csv \
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
			$(CID) \
			--conversation-id $(CID_NUM) \
			--output-prefix $(DT)-$(USR)-$(WS)ms-$(SPLIT_BY); \
	done

# create list of significant electrodes
significant-electrodes:
	mkdir -p results/sig_elecs
	python code/tfsenc_sig_elecs.py \
			--sid $(SID) \
			$(SIG) \
			--sig-thresh $(SIG_THRESH) \
			--input-directory \
				20210302-0825-ca-200ms-all-676-glove50-cnxt-1024-pca_0d-1-convo \
			--output-file-name \
				'$(DT)-$(SID)-$(EMB)-sig-elecs-1-convo'

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM);

# -----------------------------------------------------------------------------
# Plotting encoding
# -----------------------------------------------------------------------------

plot-encoding:
	mkdir -p results/enc/figures
	python code/tfsenc_plots.py \
			--sid $(SID) \
			--electrodes $(E_LIST) \
			--input-directory \
				20210301-2148-ca-200ms-all-625-glove50-cnxt-1024-pca_0d \
			--labels \
				enc-glove \
			--output-file-name \
				'$(DT)-$(SID)-glove-enc-one-conversation'

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
		--sig-elec-file \
			20210302-0855-676-glove50-sig-elecs-1-convo_comp.csv \
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
		$(CID) \
		--conversation-id $(CID_NUM) \
		--output-prefix $(DT)-$(USR)-$(WS_ERP)s-$(SPLIT_BY)-1-convo; \


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
			$(CID) \
			--conversation-id $(CID_NUM) \
			--output-prefix $(DT)-$(USR)-$(WS_ERP)s-$(SPLIT_BY); \
	done

# -----------------------------------------------------------------------------
# Plotting ERP
# -----------------------------------------------------------------------------

plot-erp:
	mkdir -p results/erp/figures
	python code/tfserp_plots.py \
			--sid $(SID) \
			$(CMB) \
			$(PROD) \
			$(COMP) \
			--electrodes $(E_LIST) \
			--input-directory \
				20210301-1946-ca-4s-correct-625-glove50-cnxt-1024-pca_0d \
				20210301-1949-ca-4s-incorrect-625-glove50-cnxt-1024-pca_0d \
				20210302-0011-ca-4s-correct-676-glove50-cnxt-1024-pca_0d \
				20210302-0004-ca-4s-incorrect-676-glove50-cnxt-1024-pca_0d \
			--labels \
				erp-comp-sig-correct-combined \
				erp-comp-sig-incorrect-combined \
			--output-file-name \
				'$(DT)-$(SID)-erp-corr-incorr-prod-sig-all-elecs-combined'


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
