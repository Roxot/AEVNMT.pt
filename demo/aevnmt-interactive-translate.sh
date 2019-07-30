export CUDA_VISIBLE_DEVICES=0
export LC_ALL=en_US.UTF-8 

USE_GPU=false
TAG=flickr
SRC=de
TGT=en
DATA=${TAG}/data
TRAINED_MODEL=${TAG}/models/aevnmt/supervised

echo "# Instructions

This is an interactive demo, when the system is 'Ready to start' you can type in (German) sentences.
The system will deal with truecasing, tokenization and word segmentation for you.
To stop use CTRL+D

Try this example sentence: Ein kleines Mädchen klettert in ein Holzhäuschen, das als Stall dient.

# AEVNMT (supervised training)

"
python -m aevnmt.translate \
    --src ${SRC} \
    --tgt ${TGT} \
    --output_dir ${TRAINED_MODEL} \
    --use_gpu ${USE_GPU} \
    --max_sentence_length 100 \
    --interactive_translation 1 \
    --split_sentences true \
    --tokenize true \
    --lowercase true \
    --bpe_codes_prefix ${DATA}/bpe_codes \
    --bpe_merge true \
    --recase true \
    --detokenize true \
    --show_raw_output false \
    --verbose true

