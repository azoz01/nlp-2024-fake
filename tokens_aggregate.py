class token_aggregate:
    def __init__(
        self,
        spacy_token,
        clean_model_tokens,
        dirty_model_tokens,
        is_spacy_NER,
        model_exp
    ):
        self.spacy_token = spacy_token
        self.clean_model_tokens = clean_model_tokens
        self.dirty_model_tokens = dirty_model_tokens
        self.is_spacy_NER = is_spacy_NER
        self.model_exp = model_exp

    def get_spacy_token(self):
        return self.spacy_token
    def get_model_tokens(self):
        return self.clean_model_tokens, self.dirty_model_tokens
    def get_model_exp(self):
        return self.model_exp

def generate_aggregate_list(doc,exp_tensor,tokens_clear,tokens_dirty):
    spacy_token_to_our_tokens = []

    spacy_tokens = []
    spacy_is_NER = []

    current_clear_token = 0
    current_spacy_token = 0


    constructed_token = ""
    clean_tokens_for_current_spacy_token = []
    dirty_tokens_for_current_spacy_token = []
    model_token_exp = []

    for token in doc:
        spacy_tokens.append(token.text)
        spacy_is_NER.append(token.ent_type_ != '')

    for token in tokens_clear:

        constructed_token = constructed_token+token
        
        clean_tokens_for_current_spacy_token.append(token)
        #weird shift
        dirty_tokens_for_current_spacy_token.append(tokens_dirty[current_clear_token+1])
        model_token_exp.append(exp_tensor[0,1,current_clear_token+1])

        if(constructed_token == spacy_tokens[current_spacy_token]):

            new_aggregate = token_aggregate(
                spacy_tokens[current_spacy_token],
                clean_tokens_for_current_spacy_token,
                dirty_tokens_for_current_spacy_token,
                spacy_is_NER[current_spacy_token],
                model_token_exp
                )
            spacy_token_to_our_tokens.append(new_aggregate)

            current_spacy_token= current_spacy_token+1

            constructed_token = ""
            clean_tokens_for_current_spacy_token = []
            dirty_tokens_for_current_spacy_token = []
            model_token_exp = []
        current_clear_token = current_clear_token+1
    return spacy_token_to_our_tokens
