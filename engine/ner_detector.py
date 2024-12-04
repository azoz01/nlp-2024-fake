import spacy
import torch
from tqdm import tqdm
from transformers import TextClassificationPipeline

from engine.tokens_aggregate import TokenAggregate
from engine.xai import FeatureAblationText

NO_NER_SYMBOL = ""
PREDICTED_CLASS = 0


def clear_tokens_from_model(tokens: list[str]) -> list[str]:
    tokens_clear = [s.replace("Ġ", "") for s in tokens]
    tokens_clear = tokens_clear[1 : len(tokens_clear) - 1]
    return tokens_clear


def tokenize_evaluate_and_detect_NERs(
    pipeline: TextClassificationPipeline,
    text: list[str],
    spacy_model: str = "en_core_web_sm",
    model_token_cleaner_function=clear_tokens_from_model,
) -> list[tuple[str, int, str]]:
    # token, exp , Ner type
    aggregates = generate_aggregates(
        pipeline, text, spacy_model, model_token_cleaner_function
    )

    token_exp_NER = []
    for aggregate in tqdm(aggregates):
        token_exp_NER += transform_aggregate_into_mapping(aggregate)
    return token_exp_NER


def generate_aggregates(
    pipeline: TextClassificationPipeline,
    text: list[str],
    spacy_model: str = "en_core_web_sm",
    model_token_cleaner_function=clear_tokens_from_model,
) -> list[TokenAggregate]:

    def forward(obs):
        return pipeline.model(obs).logits

    attr = FeatureAblationText(forward)
    tokenize_function = get_tokenizer_function(get_device())
    NER = spacy.load(spacy_model)

    model_tokens_for_texts = []
    tensors_for_attributions = []
    docs = []

    for obs in text:
        docs.append(NER(obs))

        obs_pt = tokenize_function(obs, pipeline)

        tensors_for_attributions.append(obs_pt)

        tokens = pipeline.tokenizer.convert_ids_to_tokens(obs_pt[0])
        model_tokens_for_texts.append(tokens)

    exps = []
    for tensor in tensors_for_attributions:
        exps.append(attr.get_attributions([tensor]))

    all_aggregates: list[TokenAggregate] = []
    for id, doc in enumerate(docs):
        tokens = model_tokens_for_texts[id]
        tokens_clear = model_token_cleaner_function(tokens)
        spacy_token_to_our_tokens = TokenAggregate.generate_aggregate_list(
            doc, exps[id][PREDICTED_CLASS], tokens_clear, tokens
        )
        if spacy_token_to_our_tokens is not False:
            all_aggregates = all_aggregates + spacy_token_to_our_tokens

    return all_aggregates


def transform_aggregate_into_mapping(
    aggregate: TokenAggregate,
) -> list[(str, float, str)]:
    list_: list[(str, float, str)] = []
    NER = find_NER_name_for_aggregate(aggregate.NERs)
    for id, token in enumerate(aggregate.dirty_model_tokens):
        element = (token, aggregate.model_exp[id], NER)
        list_.append(element)
    return list_


def find_NER_name_for_aggregate(NERs: list[str]) -> str:
    unique_NERs: set[str] = set(NERs) - {NO_NER_SYMBOL}
    if not unique_NERs:
        return NO_NER_SYMBOL
    elif len(unique_NERs) != 1:
        print(f"EHHH BAD ASU MOTINS: {unique_NERs}")
    return unique_NERs.pop()


def tokenize_using_cpu(
    obs: str, pipeline: TextClassificationPipeline
) -> torch.Tensor:
    return pipeline.tokenizer(obs, return_tensors="pt")["input_ids"].cpu()


def tokenize_using_cuda(
    obs: str, pipeline: TextClassificationPipeline
) -> torch.Tensor:
    return pipeline.tokenizer(obs, return_tensors="pt")["input_ids"].cuda()


def get_tokenizer_function(device: str):
    if device == "cpu":
        return tokenize_using_cpu
    else:
        return tokenize_using_cuda


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
