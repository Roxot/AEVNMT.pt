USE_GPU=true
TAG=iwslt14
SRC=de
TGT=en
DATA=${TAG}/data
CONFIG=iwslt_aevnmt_transformer_info
OUTPUT=${TAG}/models/aevnmt/${CONFIG}.${SRC}-${TGT}
HPARAMS=hparams/${CONFIG}.yml

mkdir -p ${OUTPUT}

python -m aevnmt.train \
    --hparams_file ${HPARAMS} \
    --src ${SRC} \
    --tgt ${TGT} \
    --training_prefix ${DATA}/train.bpe.32000 \
    --validation_prefix ${DATA}/valid.bpe.32000 \
    --output_dir ${OUTPUT} \
    --use_gpu ${USE_GPU} 
