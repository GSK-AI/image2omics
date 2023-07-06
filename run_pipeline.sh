#!/bin/bash

DATA_DIR=$1
SAVE_DIR=$2
RUN_INFER=$3

if ${RUN_INFER}; then
    python image2omics/preprocessing.py -d ${DATA_DIR}/data
    for OMICS in "proteomics_m1" "proteomics_m2" "transcriptomics_m1" "transcriptomics_m2";
    do
        echo ${OMICS}
        for SEED in {0..9}
        do 
            echo ${SEED}
            python image2omics/featurize.py -t ${DATA_DIR}/data/tile_manifest.txt  \
                                            -i ${DATA_DIR}/data/icf_manifest.txt \
                                            -c ${DATA_DIR}/checkpoints/${OMICS}/config.yaml \
                                            -o ${SAVE_DIR}/${OMICS} \
                                            --ckpt ${DATA_DIR}/checkpoints/${OMICS}/model_split_${SEED}.pth \
                                            -s ${SEED}
        done
        python image2omics/postprocessing.py -d ${DATA_DIR}/data/GT/${OMICS}.csv \
                                            -o ${SAVE_DIR}/${OMICS}
    done
fi

export DATA_DIR=${DATA_DIR}
export SAVE_DIR=${SAVE_DIR}
python image2omics/plotting.py --cache_folder=${DATA_DIR}/third-party --data_folder=${SAVE_DIR}/Plots

