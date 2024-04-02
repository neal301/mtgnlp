import pandas as pd

# Preprocessing

def preprocess_text(col: pd.Series):
    '''
    Takes in the 'oracle_text' column (as a series) of the 'cards' dataframe and preprocesses it.
    '''
    col = col.str.lower()
    col = col.str.replace("\n", " ")
    col = col.str.replace(r'{[^}]+}', 'symbol')
    col = col.str.replace(r'[^\w\s]', '')
    return col


# Embeddings

def bert_embedding(text):
    '''
    Takes in a string and returns the BERT embeddings of the string.
    '''
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1

    # Get the embeddings
    with torch.no_grad():
        outputs = model(input_ids)

    # outputs[0] contains the hidden states of the last layer
    # We take the embeddings from the first token of the last layer which corresponds to [CLS]
    embeddings = outputs[0][0, 0, :].numpy()

    return embeddings