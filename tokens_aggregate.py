from typing import Self
import spacy
from torch import Tensor

class TokenAggregate:
    def __init__(
        self,
        spacy_token: str,
        clean_model_tokens: list[str],
        dirty_model_tokens: list[str],
        is_spacy_NER: bool,
        model_exp: list[float]
    ) ->None:
        self.spacy_token = spacy_token
        self.clean_model_tokens = clean_model_tokens
        self.dirty_model_tokens = dirty_model_tokens
        self.is_spacy_NER = is_spacy_NER
        self.model_exp = model_exp
        
    @staticmethod
    def generate_aggregate_list(
        doc: spacy.tokens.doc.Doc,
        exp_tensor: Tensor,
        tokens_clear: list[str],
        tokens_dirty: list[str]):

        spacy_token_to_our_tokens = []

        spacy_tokens = []
        spacy_is_NER = []

        current_clear_token = 0
        current_spacy_token = 0


        constructed_token = ""
        clean_tokens_for_current_spacy_token = []
        dirty_tokens_for_current_spacy_token = []
        model_token_exp = []

        for spacy_token in doc:
            spacy_tokens.append(spacy_token.text)
            spacy_is_NER.append(spacy_token.ent_type_ != '')

        for token in tokens_clear:

            constructed_token = constructed_token+token
            
            clean_tokens_for_current_spacy_token.append(token)
            #weird shift
            dirty_tokens_for_current_spacy_token.append(tokens_dirty[current_clear_token+1])
            model_token_exp.append(float(exp_tensor[0,0,current_clear_token+1]))

            if(constructed_token == spacy_tokens[current_spacy_token]):

                new_aggregate = TokenAggregate(
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
            
        #debug!!!
        if(current_spacy_token != len(spacy_tokens)):
            print(f"\n INVALID DOC!!! stopped at {spacy_tokens[current_spacy_token]}\n")
            print(f"{constructed_token}")
            print(f"spacy tokens left: {spacy_tokens[current_spacy_token:]}")
            print(f"model tokens left: {tokens_clear[sum([len(x.clean_model_tokens) for x in spacy_token_to_our_tokens]):]}")

            for spacy_token in doc:
                if spacy_token.ent_type_:
                    print(f"Token: {spacy_token.text} True")
                else:
                    print(f"Token: {spacy_token.text} False")

        return spacy_token_to_our_tokens
