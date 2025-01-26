
# experiment settings
DATASET=cifar-100
N_CLASS=100

# save directory
OUTDIR=logs/lambda

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=5
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# query and lambda values
QUERIES=("vit" "convnext_small" "convnext_tiny" "convnext_base")  
LAMBDAS=("1e-5" "5e-5" "1e-4" "5e-4")

# loop through queries and lambdas
for QUERY in "${QUERIES[@]}"; do
    for LAMBDA in "${LAMBDAS[@]}"; do
        python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
            --learner_type prompt --learner_name OSPrompt \
            --prompt_param 100 8 $LAMBDA --query $QUERY \
            --log_dir ${OUTDIR}/${QUERY}/lambda-${LAMBDA}
    done
done