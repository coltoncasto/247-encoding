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
# E_LIST := 1 20 72

# 676 Electrode IDs
E_LIST := $(shell seq 1 1)
# E_LIST := $(shell seq 1 63)

# Choose the subject to run for
SID := 625
# SID := 676

NPERM := 1
# NPERM := 10

# Choose the lags to run for.
LAGS := {-5000..5000..25}
LAGS := {-2000..2000..25}
# LAGS := {-62000..-58000..25}
# LAGS := 0

# Choose which set of embeddings to use
EMB := glove50
# EMB := gpt2-xl

# Choose whether to remove words without glove embeddings
GLV := --remove-glove

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
# SPLIT := --split-flag
SPLIT_BY := all
# SPLIT_BY := correct
# SPLIT_BY := incorrect
# SPLIT_BY := incorrect-w-predicted
# SPLIT_BY := most-frequent
# SPLIT_BY := least-frequent

# Choose significant electrodes based on threshold
SIG := --sig-thresh-flag
SIG := --sig-phase-shuffle-flag
SIG_THRESH := .12 # correlation used w/ thresh
SIG_THRESH := .01 # alpha used w/ phase shuffle

# take intersection of significant electrodes with other list
# INT = --intersection

# prod/comp flags for plotting
PROD := --prod
COMP := --comp

# combine subjects for plotting 
# e.g., take four directors and combine 1/3 and 2/4
# CMB := --combine-subjects

# only look at one conversation for significant encodings
# CID := --conversation-id-flag
CID_NUM := 5
# CID_NUM := 63
# CID_NUM := 53
CID_NUM := 35

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
#		--sig-elec-file 
#			sig_elecs_prod_test.csv 

run-encoding:
	mkdir -p logs
	$(CMD) code/$(FILE).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		$(GLV) \
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
		--output-prefix $(DT)-$(USR)-$(WS)ms-$(SPLIT_BY)-harsha-test; \


# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
# 			--sig-elec-file
#				20210302-0855-676-glove50-sig-elecs-1-convo_comp.csv 
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--sid $(SID) \
			--electrodes $$elec \
			--emb-type $(EMB) \
			$(GLV) \
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
			$(CMB) \
			$(PROD) \
			$(COMP) \
			--electrodes $(E_LIST) \
			--input-directory \
				20210415-1528-ca-200ms-all-convo-5-new-load-remove-gpt2-split-625-glove50-cnxt-1024-pca_0d \
			--labels \
				correct \
				incorrect-actual \
			--output-file-name \
				'$(DT)-$(SID)-enc-test'

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
			sig_elecs_prod.csv \
		--emb-type $(EMB) \
		$(GLV) \
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
		--output-prefix $(DT)-$(USR)-$(WS_ERP)s-$(SPLIT_BY)-codebase-meeting-test; \


# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-erp-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE_ERP).py \
			--sid $(SID) \
			--electrodes $$elec \
			--emb-type $(EMB) \
			$(GLV) \
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
				20210417-1230-ca-4s-correct-comp-thesis-625-glove50-cnxt-1024-pca_0d-subset \
				20210417-1237-ca-4s-incorrect-comp-thesis-625-glove50-cnxt-1024-pca_0d-subset \
				20210417-1233-ca-4s-correct-comp-thesis-676-glove50-cnxt-1024-pca_0d-subset \
				20210417-1251-ca-4s-incorrect-comp-thesis-676-glove50-cnxt-1024-pca_0d-subset \
			--labels \
				correctly \
				incorrectly \
			--output-file-name \
				'$(DT)-erp-comp-thesis-subset-no-label-sig-test'

# -----------------------------------------------------------------------------
#  Significant electrodes
# -----------------------------------------------------------------------------

# create list of significant electrodes
significant-electrodes:
	mkdir -p results/sig-elecs
	python code/tfsenc_sig_elecs.py \
			--sid $(SID) \
			$(SIG) \
			--sig-thresh $(SIG_THRESH) \
			$(INT) \
			--intersection-file-comp \
				sig_elecs_comp.csv \
			--intersection-file-prod \
				sig_elecs_prod.csv \
			--input-directory \
				harsha-test-625-1-convo-5 \
			--output-file-name \
				'$(DT)-$(SID)-$(EMB)-sig-elecs-thesis'

# -----------------------------------------------------------------------------
#  Misc. targets
# -----------------------------------------------------------------------------

# If you have pickled the data yourself, then you can just link to it
link-data:
	ln -fs $(shell dirname `pwd`)/247-pickling/results/* data/

link-data-shuffled:
	ln -fs /scratch/gpfs/hgazula/247-encoding/results/tfs/colton-phase-shuffle/* results/sig-elecs/phase-shuffle/

sync-plots:
	rsync -aPv --delete \
	    results/figures/ \
	    /tigress/$(USER)/247-encoding-results/figures

sync-results:
	rsync -aP --delete \
	    results/ \
	    /tigress/$(USER)/247-encoding-results
