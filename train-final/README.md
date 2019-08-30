# Train ConvNet on ImageNet

## Train ConvNet on ImageNet 

To train the found ConvNet (the result of [nas-search/](/nas-search/)), 
first the NAS-decisions (ConvNet encoding) are parsed from the 
output_dir of the NAS search (defined with the --parse_search_dir) flag. 
Then the parsed ConvNet arch is trained for 350 epochs on ImageNet. 
The training follows the MnasNet training schedule and hyper-parameters
from the original MnasNet-TPU repo.

### Specific steps

1. Setting up ImageNet dataset

To setup the ImageNet follow the instructions from [here](https://cloud.google.com/tpu/docs/tutorials/amoebanet#full-dataset)  


2. Setting up ENV variables:
```
export DATA_DIR=${imagenet-dataset-location}
export OUTPUT_DIR=${output-location}/model-single-path-train-final
export PARSE_DIR=${output-location}/model-single-path-search
export CUDA_VISIBLE_DEVICES=${gpu-ids}
```

3. Launch training:
```
lambda_val=0.020; python main.py --data_dir=$DATA_DIR --model_dir=${OUTPUT_DIR}/lambda-val-${lambda_val}/ --parse_search_dir=${PARSE_DIR}/lambda-val-${lambda_val}/

```

