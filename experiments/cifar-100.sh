# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=100

# save directory
OUTDIR=DATA_PATH_USER/${DATASET}/10-task

# hard coded inputs
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=5
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR


# OS-P++
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = qr loss weight
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4 --query vit\
    --log_dir ${OUTDIR}/os-p++




# OS-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = qr loss weight
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 0.0 --query vit\
    --log_dir ${OUTDIR}/os-p


# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 \
    --log_dir ${OUTDIR}/dual-prompt

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 1 \
    --log_dir ${OUTDIR}/deepl2p

python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir ${OUTDIR}/l2p