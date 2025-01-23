# NLP project

## Model training
* Install dependencies - `pip install -r requirements.txt`
* Download required spacy model - `python -m spacy download en_core_web_lg`
* Training all models - `./train_models.sh`
* Training output should be in `output` directory


## Reproduction of results
* Finish model training - this should result in a folder output with models partitioned by dataset, model, training type and run number like:
```
output
    | - coaid
        | - ernie
            | - masked
                | - 1
                    | - test_acc.json
                    | - model_final
                        | - model.safetensors
```
* Install dependencies - if model training was done elsewhere
* run notebooks/results.ipynb - to obtain raw results in json
* change the utilised json results in results_analysis.ipynb
* run notebooks/results_analysis.ipynb - to obtain the figures