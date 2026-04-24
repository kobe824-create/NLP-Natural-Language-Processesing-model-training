import os
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Optional: suppress tokenizer parallelism warnings ──────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════════════════════
#  SLANG / ABBREVIATION NORMALISER
#  Expands common internet slang before the model sees the text so that
#  "omg this slaps fr fr" is understood the same way as
#  "oh my god this is great, for real".
# ══════════════════════════════════════════════════════════════════════════════
SLANG_MAP = {
    r"\bomg\b": "oh my god",
    r"\blol\b": "laughing out loud",
    r"\blmao\b": "laughing my ass off",
    r"\blmfao\b": "laughing my fucking ass off",
    r"\brotfl\b": "rolling on the floor laughing",
    r"\bwtf\b": "what the fuck",
    r"\bwth\b": "what the heck",
    r"\bidk\b": "i don't know",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my honest opinion",
    r"\bngl\b": "not gonna lie",
    r"\btbh\b": "to be honest",
    r"\bfr\b": "for real",
    r"\bfr fr\b": "for real for real",
    r"\bngl\b": "not gonna lie",
    r"\bsmh\b": "shaking my head",
    r"\bfml\b": "fuck my life",
    r"\bgoat\b": "greatest of all time",
    r"\blit\b": "amazing",
    r"\bslaps\b": "is great",
    r"\bfire\b": "amazing",
    r"\bpog\b": "impressive",
    r"\bpoggers\b": "very impressive",
    r"\bslay\b": "excellent",
    r"\bbased\b": "admirable",
    r"\bcope\b": "deal with disappointment",
    r"\bseethe\b": "be very angry",
    r"\bcringe\b": "embarrassing",
    r"\bw\b": "win",
    r"\bl\b": "loss",
    r"\bnpc\b": "boring person",
    r"\bgg\b": "good game",
    r"\bgg ez\b": "easy win",
    r"\bno cap\b": "no lie",
    r"\bcap\b": "lie",
    r"\bhit different\b": "feels special",
    r"\bvibe\b": "feeling",
    r"\bvibes\b": "feelings",
    r"\bglow up\b": "improve greatly",
    r"\bflex\b": "show off",
    r"\bbussin\b": "very good",
    r"\bsuss\b": "suspicious",
    r"\bamong us\b": "suspicious",
    r"\bclout\b": "fame",
    r"\bstan\b": "obsessive fan",
    r"\bshipp(ing|ed)?\b": "supporting a relationship",
    r"\bthrowback\b": "memory of the past",
    r"\bbig yikes\b": "very embarrassing",
    r"\byikes\b": "embarrassing",
    r"\bnope\b": "no",
    r"\bnah\b": "no",
    r"\byeah\b": "yes",
    r"\byep\b": "yes",
    r"\bbet\b": "agreed",
    r"\bfacts\b": "that is true",
    r"\bpreach\b": "i agree strongly",
    r"\bdead\b": "i find this hilarious",
    r"\bi'm dead\b": "i find this extremely funny",
    r"\bsalty\b": "bitter or upset",
    r"\bbitter\b": "resentful",
    r"\btoxic\b": "harmful behaviour",
    r"\bpetty\b": "unnecessarily mean",
    r"\bextra\b": "over the top",
    r"\bthirsty\b": "desperate",
    r"\bsavage\b": "brutally honest or impressive",
    r"\bbaddie\b": "attractive confident person",
    r"\bfaded\b": "intoxicated",
    r"\bwack\b": "bad or weird",
    r"\blegit\b": "genuine",
    r"\bwoke\b": "socially aware",
    r"\bkaren\b": "entitled complaining person",
    r"\bboomer\b": "out of touch older person",
    r"\bzoomer\b": "young person",
    r"\bsimp\b": "overly submissive person",
    r"\bmid\b": "mediocre",
    r"\btrash\b": "very bad",
    r"\bdogwater\b": "extremely bad",
    r"\bgigachad\b": "extremely impressive person",
    r"\bchad\b": "impressive confident person",
    r"\bop\b": "overpowered",
    r"\bnerf\b": "make weaker",
    r"\bbuff\b": "make stronger",
    r"\bhype\b": "excitement",
    r"\bhyped\b": "very excited",
    r"\bclapped\b": "beaten badly",
    r"\bsweaty\b": "trying too hard",
    r"\bnoob\b": "beginner",
    r"\bn00b\b": "beginner",
    r"\btroll\b": "provocateur",
    r"\bflame\b": "harsh criticism",
    r"\broast\b": "harsh jokes",
    r"\bdiss\b": "disrespect",
    r"\bclout chaser\b": "someone seeking attention",
    r"\bcooked\b": "in serious trouble",
    r"\btouched\b": "crazy",
    r"\bmenace\b": "troublemaker",
    r"\brat\b": "traitor",
    r"\bsnek\b": "traitor",
    r"\bfake\b": "not genuine",
    r"\bclone\b": "copy",
    r"\bdrip\b": "stylish outfit",
    r"\bfit\b": "outfit",
    r"\bfit check\b": "show me your outfit",
    r"\bslept on\b": "underrated",
    r"\boverhyped\b": "not as good as expected",
    r"\bunderrated\b": "better than people think",
    r"\bbreaking\b": "very funny or surprising",
    r"\bscam\b": "fraud",
    r"\bbait\b": "trap",
    r"\bfood for thought\b": "something to consider",
    r"\bpainful\b": "very bad",
    r"\bhurts\b": "is painful",
    r"\bsmashed it\b": "did very well",
    r"\bnailed it\b": "succeeded perfectly",
    r"\bkilled it\b": "performed excellently",
    r"\bcrushed it\b": "performed excellently",
    r"\bbooked and blessed\b": "very successful",
    r"\bwinning\b": "succeeding",
    r"\bballing\b": "living lavishly",
    r"\bdripping\b": "wearing stylish clothes",
    r"\bswerve\b": "avoid",
    r"\bghostd?\b": "ignored without explanation",
    r"\bleft on read\b": "message seen but not replied to",
    r"\bblocked\b": "prevented from contact",
    r"\bcancel(led|ed)?\b": "publicly denounced",
    r"\bcancelled\b": "publicly denounced",
    r"\bcancel culture\b": "public denunciation movement",
    r"\baired\b": "ignored",
    r"\bvibe check\b": "assess the mood",
    r"\bchill\b": "relaxed",
    r"\blow key\b": "secretly or quietly",
    r"\bhigh key\b": "very much or openly",
    r"\bnot it\b": "not good or appropriate",
    r"\bthats it\b": "that is perfect",
    r"\bthats the one\b": "that is perfect",
    r"\bon god\b": "i swear",
    r"\bon gang\b": "i swear",
    r"\bword\b": "i agree",
    r"\bsaying\b": "claiming",
    r"\bno\?\b": "isn't it",
    r"\binnit\b": "isn't it",
    r"\binit\b": "isn't it",
    r"\bbruh\b": "expression of disbelief",
    r"\bbro\b": "friend",
    r"\bdude\b": "friend",
    r"\bfam\b": "family or close friend",
    r"\bsquad\b": "friend group",
    r"\bgang\b": "friend group",
    r"\bhomie\b": "close friend",
    r"\bmate\b": "friend",
    r"\bblud\b": "friend",
    r"\bmandem\b": "group of friends",
    r"\bwagwan\b": "what is going on",
    r"\bpeng\b": "attractive or good",
    r"\bbless\b": "thank you or good",
    r"\bbread\b": "money",
    r"\bdough\b": "money",
    r"\bcheddar\b": "money",
    r"\bscrilla\b": "money",
    r"\bguap\b": "money",
    r"\bdough\b": "money",
    r"\bpaper\b": "money",
    r"\bstacks\b": "large amount of money",
    r"\bbands\b": "thousands of dollars",
    r"\bracks\b": "thousands of dollars",
    r"\bcoin\b": "money",
    r"\bmoolah\b": "money",
    r"\bloot\b": "money or prizes",
    r"\bswag\b": "style or free stuff",
    r"\bgear\b": "equipment or drugs",
    r"\bstuff\b": "things",
    r"\bgoons\b": "dangerous associates",
    r"\bhench\b": "muscular",
    r"\bbuff\b": "attractive or muscular",
    r"\bgassed\b": "very excited or praised",
    r"\bhype beast\b": "someone obsessed with trends",
    r"\bflex on\b": "show off to",
    r"\bcap on\b": "lie to",
    r"\bpull up\b": "arrive or confront",
    r"\bdip\b": "leave",
    r"\bpeace\b": "goodbye",
    r"\bdeuces\b": "goodbye",
    r"\bcatch you later\b": "goodbye",
    r"\blmk\b": "let me know",
    r"\bnvm\b": "never mind",
    r"\bbtw\b": "by the way",
    r"\bfyi\b": "for your information",
    r"\bafaik\b": "as far as i know",
    r"\bafk\b": "away from keyboard",
    r"\bbrb\b": "be right back",
    r"\bttyl\b": "talk to you later",
    r"\bttys\b": "talk to you soon",
    r"\bidc\b": "i don't care",
    r"\bibf\b": "internet best friend",
    r"\birl\b": "in real life",
    r"\bfomo\b": "fear of missing out",
    r"\bjomo\b": "joy of missing out",
    r"\btmi\b": "too much information",
    r"\bsmfh\b": "shaking my fucking head",
    r"\bffs\b": "for fuck's sake",
    r"\bomfg\b": "oh my fucking god",
    r"\bstfu\b": "shut the fuck up",
    r"\bgtfo\b": "get the fuck out",
    r"\bgtg\b": "got to go",
    r"\bg2g\b": "got to go",
    r"\bigtg\b": "i got to go",
}

def normalise_slang(text: str) -> str:
    """Lower-case and expand slang/abbreviations."""
    text = text.lower()
    for pattern, replacement in SLANG_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
#  Downloads real-world sentiment data from Hugging Face Hub at runtime.
#  No giant files committed to your repo — just a pip install.
# ══════════════════════════════════════════════════════════════════════════════
def load_online_data(max_per_class: int = 5000):
    """
    Load sentiment data from Hugging Face datasets.

    Datasets used:
      1. stanfordnlp/sst2   — Stanford Sentiment Treebank (binary: positive / negative)
      2. tweet_eval/sentiment — Twitter data (positive / neutral / negative) — rich in slang

    Returns (texts, labels) where labels follow: 1=positive, 0=neutral, -1=negative.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install datasets"
        )

    texts, labels = [], []

    # ── 1. SST-2 (binary sentiment) ───────────────────────────────────────────
    print("Downloading SST-2 dataset …")
    sst = load_dataset("stanfordnlp/sst2", split="train")
    pos_count = neg_count = 0
    for row in sst:
        sentence = row["sentence"].strip()
        label_val = row["label"]   # 1 = positive, 0 = negative in SST-2
        if label_val == 1 and pos_count < max_per_class:
            texts.append(sentence)
            labels.append(1)
            pos_count += 1
        elif label_val == 0 and neg_count < max_per_class:
            texts.append(sentence)
            labels.append(-1)
            neg_count += 1
        if pos_count >= max_per_class and neg_count >= max_per_class:
            break
    print(f"  SST-2: {pos_count} positive, {neg_count} negative loaded.")

    # ── 2. TweetEval / Sentiment (ternary, includes slang) ────────────────────
    print("Downloading TweetEval-sentiment dataset …")
    tweet = load_dataset("tweet_eval", "sentiment", split="train", trust_remote_code=True)
    tp = tn = tne = 0
    for row in tweet:
        sentence = row["text"].strip()
        lbl = row["label"]   # 0=negative, 1=neutral, 2=positive
        if lbl == 2 and tp < max_per_class:
            texts.append(sentence)
            labels.append(1)
            tp += 1
        elif lbl == 1 and tn < max_per_class:
            texts.append(sentence)
            labels.append(0)
            tn += 1
        elif lbl == 0 and tne < max_per_class:
            texts.append(sentence)
            labels.append(-1)
            tne += 1
        if tp >= max_per_class and tn >= max_per_class and tne >= max_per_class:
            break
    print(f"  TweetEval: {tp} positive, {tn} neutral, {tne} negative loaded.")

    return texts, labels


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_sentiment_model(max_per_class: int = 5000):
    print("\n=== Sentiment Model Training ===\n")

    # 1. Load data
    raw_texts, labels = load_online_data(max_per_class=max_per_class)

    # 2. Normalise slang
    print("\nNormalising slang …")
    texts = [normalise_slang(t) for t in raw_texts]

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 4. Pipeline: TF-IDF (char + word ngrams) → Logistic Regression
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 3),        # unigrams, bigrams, trigrams
                analyzer="word",
                sublinear_tf=True,         # apply log(tf) — helps with skewed freqs
                min_df=2,                  # ignore very rare tokens
                max_features=80_000,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=1.5,
                max_iter=1000,
                solver="saga",
                multi_class="multinomial",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nFitting model …")
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    print("\n── Test-set results ──")
    print(classification_report(
        y_test, y_pred,
        target_names=["Negative (-1)", "Neutral (0)", "Positive (1)"],
        labels=[-1, 0, 1],
    ))

    # 6. Save
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/nlp_model.pkl")
    print("Model saved → model/nlp_model.pkl")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK DEMO
# ══════════════════════════════════════════════════════════════════════════════
LABEL_MAP = {1: "Positive 😊", 0: "Neutral 😐", -1: "Negative 😞"}

def predict(model, sentences):
    normalised = [normalise_slang(s) for s in sentences]
    preds = model.predict(normalised)
    probs = model.predict_proba(normalised)
    classes = model.classes_.tolist()

    print("\n── Predictions ──")
    for sentence, pred, prob in zip(sentences, preds, probs):
        confidence = prob[classes.index(pred)]
        print(f"  [{LABEL_MAP[pred]:14s}  {confidence:.0%}]  {sentence}")


if __name__ == "__main__":
    trained_model = train_sentiment_model(max_per_class=5000)

    demo_sentences = [
        # Standard
        "I absolutely love this product, it's amazing!",
        "This is the worst thing I have ever bought.",
        "It's okay, nothing special.",
        # Slang / internet speak
        "omg this slaps fr fr no cap 🔥",
        "bruh this is mid tbh ngl",
        "lmao bro this is absolute dogwater, don't waste your money",
        "gg ez, totally bussin, goat product ngl",
        "yikes this is cringe af smh",
        "lowkey kinda suss but idk bet",
        "bro this thing is cooked fr, such a scam",
        "ngl this hits different, based product",
        "this drip is hard, slay bestie",
        "this is painfully mid, wack af",
        "wagwan, this ting is peng no cap fam",
    ]

    predict(trained_model, demo_sentences)