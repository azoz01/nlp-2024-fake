# NLP project

## TODO

* (AZ) provide all transformers (on org data + on data with swapped tokens) (if not already done)
* (AZ) decide about using LLM
* (MR) implement counterfactuals heuristics based on the notebook `poc.ipynb`
    * assign for each person-token its attribution
    * for each observation, indentify person-tokens and the mdoel's prediction
    * if pred is `1`, then look for the other person-token with the most negative attribution (most positive for `0`)
    * try, let's say, top 10 token to replace and check if the prediction is swapped
    * if the observation contain multiple person-tokens, do it for all of them
    * for each observation, save and plot how many prediction swaps was done
    * for each person-token, save and plot how many prediction swaps was done
    * provide a mapping (token-person) x (swap success rate)
    * do this on both models for each dataset
    * we should have at the end some statistics (e.g. avg success rate), some plots (of statistics) and randomly-or-by-hand chosen examples (5-10, up to you)
* (MR) calculate attribution per type of named entity per dataset (already done in `xai_example.ipynb` so you need only to change it to a script and make better plot)
* (MR) do the similar to the previous one but per token-person and calculate some statistics/plots (we try to answer the question, if there is a group of token-persons that has significantly bigger attribution than others, if there is a big variance, what is the maginitude etc)
