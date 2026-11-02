from transformers import MistralCommonBackend

def get_tokens_from_convo(conversation, tokenizer: MistralCommonBackend):
    # A conversation is a list of dicts
    # each dict has two keys: "content" which is a free form str
    # and "role" which is a str out of "assistant" or "user"
    return tokenizer.apply_chat_template(conversation, tokenize=True, return_dict=False)