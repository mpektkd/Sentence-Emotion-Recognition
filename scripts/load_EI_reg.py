import glob
import html
import os

from config import DATA_PATH

SEPARATOR = "\t"


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Args:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        columns = line.rstrip().split(SEPARATOR)
        tweet_id = columns[0]
        text = columns[1:]
        text = clean_text(" ".join(text))
        sentiment = columns[2]
        score = columns[3]
        data[tweet_id] = (text, sentiment, score)
    return data


def load_from_dir(path, emotion=None):
    if emotion is None:
        files = glob.glob(path + "/*.txt", recursive=True)
    else:
        files = [path]
    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values())


def load_EI(emotion=None):

    if emotion is None:
        train = load_from_dir(os.path.join(DATA_PATH, "EI-reg-En-train"), emotion=emotion)
        dev = load_from_dir(os.path.join(DATA_PATH, "2018-EI-reg-En-dev"), emotion=emotion)
    
    else:
        train = load_from_dir(os.path.join(DATA_PATH, "EI-reg-En-train/EI-reg-En-"+emotion+"-train.txt"), emotion=emotion)
        dev = load_from_dir(os.path.join(DATA_PATH, "2018-EI-reg-En-dev/2018-EI-reg-En-"+emotion+"-dev.txt"), emotion=emotion)
    

    X_train = [x[0] for x in train]
    y_train = [x[1] for x in train]
    z_train = [x[2] for x in train]
    X_dev = [x[0] for x in dev]
    y_dev = [x[1] for x in dev]
    z_dev = [x[2] for x in dev]

    return X_train, y_train, z_train, X_dev, y_dev, z_dev

