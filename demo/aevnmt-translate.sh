export CUDA_VISIBLE_DEVICES=0
export LC_ALL=en_US.UTF-8 

USE_GPU=false
TAG=flickr
SRC=de
TGT=en
DATA=${TAG}/data
TRAINED_MODEL=${TAG}/models/aevnmt/supervised.${SRC}-${TGT}

python -m aevnmt.translate \
    --src ${SRC} \
    --tgt ${TGT} \
    --output_dir ${TRAINED_MODEL} \
    --translation_input_file ${DATA}/dev.${SRC} \
    --translation_output_file ${TRAINED_MODEL}/dev.${TGT}-translation \
    --translation_ref_file ${DATA}/dev.${TGT} \
    --use_gpu ${USE_GPU} \
    --max_sentence_length -1 \
    --split_sentences false \
    --detokenize true \
    --recase false \
    --bpe_merge true \
    --postprocess_ref true \
    --verbose true

