import os
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

DEVICE = "cuda"
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Base")
model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Base")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_dataset(dataset_dir: str = r'./AFND/Dataset') -> np.array:
    """
    Reads the AFND dataset from your folder layout and returns a numpy array
    with columns [Title, Text, Target].
    """

    sources_file_path = r'./AFND/sources.json'
    with open(sources_file_path, 'r', encoding='utf-8') as sources_file:
        sources_data = json.load(sources_file)
    sources_df = pd.DataFrame(list(sources_data.items()), columns=['source', 'label'])

    articles_data = []
    for source in sources_df['source']:
        scraped_articles_path = os.path.join(dataset_dir, source, 'scraped_articles.json')
        if os.path.exists(scraped_articles_path):
            with open(scraped_articles_path, 'r', encoding='utf-8') as articles_file:
                source_articles_dict = json.load(articles_file)
                source_articles_list = source_articles_dict.get('articles', [])
                for article in source_articles_list:
                    article['source'] = source
                articles_data.extend(source_articles_list)
    articles_df = pd.DataFrame(articles_data)
    
    mnerged_df = pd.merge(articles_df, sources_df, how='inner', left_on='source', right_on='source')

    cols_to_drop = [c for c in ['published date', 'source'] if c in mnerged_df.columns]
    news = mnerged_df.drop(columns=cols_to_drop)

    binary_news = news[news['label'] != 'undecided']

    num_credible = len(binary_news[binary_news['label'] == 'credible'])
    num_not_credible = len(binary_news[binary_news['label'] == 'not credible'])

    print(f"{num_credible} credible articles have been read successfully")
    print(f"{num_not_credible} not credible articles have been read successfully")

    title_col_candidates = ["title", "Title"]
    text_col_candidates = ["text", "Text", "content", "body"]
    title_col = next((c for c in title_col_candidates if c in binary_news.columns), None)
    text_col = next((c for c in text_col_candidates if c in binary_news.columns), None)
    if title_col is None or text_col is None:
        raise ValueError(f"Could not find title/text columns. Available: {list(binary_news.columns)}")

    out_df = binary_news[[title_col, text_col, "label"]].rename(
        columns={title_col: "Title", text_col: "Text", "label": "Target"}
    )

    return np.array(out_df)

def split_and_preprocess(dataset: np.array, max_len: int = 350) -> tuple[list]:
    """
    Splits the dataset, optionally filters long texts by word count,
    and returns: X_train_title, X_test_title, X_train_text, X_test_text, y_train, y_test
    """
    titles = dataset[:, 0]
    texts = dataset[:, 1]
    target = dataset[:, 2]

    df_titles = pd.DataFrame(titles, columns=['Title'])
    df_text = pd.DataFrame(texts, columns=['Text'])
    df_target = pd.DataFrame(target, columns=['Target'])

    dataset_df = pd.concat([df_titles, df_text, df_target], axis=1)

    # keep rows where Text has <= max_len words
    dataset_df = dataset_df[dataset_df['Text'].apply(lambda x: isinstance(x, str) and len(x.split()) <= max_len)]

    X = dataset_df[['Title', 'Text']]
    y = dataset_df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # NOTE: With Falcon we do NOT apply AraBERT-specific preprocessing.
    # If you still want light normalization (e.g., strip), do it here:
    def _clean(s):
        return s if isinstance(s, str) else ""
    X_train_title = [ _clean(t) for t in X_train['Title'] ]
    X_test_title  = [ _clean(t) for t in X_test['Title'] ]
    X_train_text  = [ _clean(t) for t in X_train['Text'] ]
    X_test_text   = [ _clean(t) for t in X_test['Text'] ]

    return X_train_title, X_test_title, X_train_text, X_test_text, y_train, y_test

def tokenize_text(texts, max_size: int):
    """
    Tokenizes with Falcon tokenizer. Keeps your original padding/truncation style
    and returns tensors on the selected device.
    """
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        max_length=max_size,
    ).to(DEVICE)

if __name__ == '__main__':
    # read the dataset
    binary_news = read_dataset(dataset_dir=r'.\AFND\Dataset')

    X_train_title, X_test_title, X_train_text, X_test_text, y_train, y_test = split_and_preprocess(
        dataset=binary_news, max_len=350
    )

    # tokenize (keeps your original max lengths)
    X_train_title_tokenize = tokenize_text(X_train_title, max_size=20)
    X_train_text_tokenize  = tokenize_text(X_train_text,  max_size=280)
    X_test_title_tokenize  = tokenize_text(X_test_title,  max_size=20)
    X_test_text_tokenize   = tokenize_text(X_test_text,   max_size=280)

    # create output dir
    os.makedirs('Split', exist_ok=True)

    # save tensors (same filenames as before)
    torch.save(X_train_title_tokenize, "./Split/X_train_title_tokenize.pt")
    torch.save(X_test_title_tokenize,  "./Split/X_test_title_tokenize.pt")
    torch.save(X_train_text_tokenize,  "./Split/X_train_text_tokenize.pt")
    torch.save(X_test_text_tokenize,   "./Split/X_test_text_tokenize.pt")
    torch.save(y_train,                "./Split/y_train.pt")
    torch.save(y_test,                 "./Split/y_test.pt")

    print('Saved the training and testing data successfully!')