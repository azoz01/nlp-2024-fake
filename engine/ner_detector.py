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

def text_preprocess(text: list[str]) -> list[str]:
    return [s.strip() for s in text]


def tokenize_evaluate_and_detect_NERs(
    pipeline: TextClassificationPipeline,
    text: list[str],
    spacy_model: str = "en_core_web_sm",
    model_token_cleaner_function=clear_tokens_from_model,
    return_clear_tokens: bool = False,
    ners_to_calculate_ablation: list[str] = None
) -> list[tuple[str, int, str]]:
    # token, exp , Ner type
    text = text_preprocess(text)
    masks = None
    if ners_to_calculate_ablation:
        masks = generate_masks(pipeline,text,spacy_model,model_token_cleaner_function, ners_to_calculate_ablation=ners_to_calculate_ablation)

    aggregates = generate_aggregates(
        pipeline, text, spacy_model, model_token_cleaner_function,
        NER_masks=masks
    )

    token_exp_NER = []
    for doc_aggregate in tqdm(aggregates):
        token_exp_NER_for_doc = []
        for aggregate in doc_aggregate:
            token_exp_NER_for_doc += transform_aggregate_into_mapping(aggregate, return_clear_tokens)
        token_exp_NER.append(token_exp_NER_for_doc)

    token_exp_NER_merged = [item for sublist in token_exp_NER for item in sublist]

    return token_exp_NER_merged

def generate_masks(
    pipeline: TextClassificationPipeline,
    text: list[str],
    spacy_model: str = "en_core_web_sm",
    model_token_cleaner_function=clear_tokens_from_model,
    return_clear_tokens: bool = False,
    ners_to_calculate_ablation: list[str] = None):

    aggregates = generate_aggregates(
        pipeline, text, spacy_model, model_token_cleaner_function,evaluate=False)
        
    token_exp_NER = []
    for doc_aggregate in tqdm(aggregates):
        token_exp_NER_for_doc = []
        for aggregate in doc_aggregate:
            token_exp_NER_for_doc += transform_aggregate_into_mapping(aggregate, return_clear_tokens)
        token_exp_NER.append(token_exp_NER_for_doc)

    masks = from_aggregate_list_create_mask(ners_to_calculate_ablation,token_exp_NER)

    return masks
    

def from_aggregate_list_create_mask(ners_to_calculate_ablation: list[str],token_exp_NER: list[list[tuple[str, int, str]]]):
    masks = []
    for doc_token_exp_NER in token_exp_NER:
        mask_for_doc = []
        for (token,exp,NER) in doc_token_exp_NER:
            if NER in ners_to_calculate_ablation:
                mask_for_doc.append(1)
            else:
                mask_for_doc.append(0)
        masks.append(mask_for_doc)
    return masks


def generate_aggregates(
    pipeline: TextClassificationPipeline,
    text: list[str],
    spacy_model: str = "en_core_web_sm",
    model_token_cleaner_function=clear_tokens_from_model,
    evaluate = True,
    NER_masks = None,
) -> list[list[TokenAggregate]]:

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
    for i,tensor in enumerate(tensors_for_attributions):
        if(evaluate):
            if(NER_masks):
                mask = NER_masks[i]
                try:
                    exps.append(attr.get_grouped_attribution([tensor],[torch.tensor([0]+mask+[0])]))
                except:
                    print(i)
                    print(text[i])
            else:
              exps.append(attr.get_attributions([tensor]))  
        else:
            exps.append([torch.zeros_like(tensor)])

    all_aggregates: list[list[TokenAggregate]] = []
    for id, doc in enumerate(docs):
        tokens = model_tokens_for_texts[id]
        tokens_clear = model_token_cleaner_function(tokens)
        spacy_token_to_our_tokens = TokenAggregate.generate_aggregate_list(
            doc, exps[id][PREDICTED_CLASS], tokens_clear, tokens
        )
        if spacy_token_to_our_tokens is not False:
            all_aggregates.append(spacy_token_to_our_tokens)

    return all_aggregates


def transform_aggregate_into_mapping(
    aggregate: TokenAggregate,
    take_clear_tokens: bool = False,
) -> list[tuple[str, int, str]]:
    list_: list[tuple[str, int, str]] = []
    NER = find_NER_name_for_aggregate(aggregate.NERs)
    tokens = aggregate.get_tokens(get_clear=take_clear_tokens)
    for id, token in enumerate(tokens):
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
