# Bank Jago Review Sentiment Analysis

Semester 6 NLP assignment. Full pipeline from raw Google Play Store data to trained sentiment classifiers, covering both Indonesian and English reviews.

---

## Situation

Bank Jago has tens of thousands of reviews on Google Play Store written in both Indonesian and English. The sentiment is genuinely mixed: users complain about specific things like failed transfers and login errors, and they praise specific things like free transfer quotas and the Pocket feature. That mix makes it more useful as a classification task than picking an app where 90% of reviews are five stars.

---

## Task

Build a complete sentiment analysis pipeline: scrape, preprocess, analyze, classify. Indonesian and English reviews needed separate preprocessing pipelines because slang normalization and stemming are language-specific. The deliverable is a set of trained models that can classify new reviews by sentiment.

---

## Action

### Data Collection

Scraped 75,854 reviews from `com.jago.digitalBanking` using `google_play_scraper`. Both language tracks were pulled from the Indonesian store (`country=id`). Output is one CSV with a `lang` column.

| Language | Reviews |
|----------|---------|
| Indonesian (`lang=id`) | 67,358 |
| English (`lang=en`) | 8,496 |
| Total | 75,854 |

### Preprocessing

Indonesian reviews: lowercase, remove punctuation and URLs, normalize slang (`gak` to `tidak`, `kl` to `kalau`), tokenize with NLTK, remove stopwords, Sastrawi stemming.

English reviews: same basic cleaning steps, no slang dictionary or Sastrawi.

Sentiment labels assigned by star rating:

| Score | Label |
|-------|-------|
| 1 to 3 | negative / negatif |
| 4 to 5 | positive / positif |

Each language saved to its own preprocessed CSV.

### EDA

Both language notebooks cover: sentiment distribution, star rating breakdown, review length analysis, polarity scatter plots, top word frequency, wordclouds, bigram analysis, and correlation heatmaps. The English notebook includes a `langdetect` filter to remove Indonesian reviews that slipped through the English track.

### Model Training

TF-IDF vectorization (max 10,000 features), 80/20 stratified train-test split. Three classifiers trained per language:

| Model | Implementation |
|-------|---------------|
| Logistic Regression | `LogisticRegression(max_iter=1000)` |
| Naive Bayes | `MultinomialNB()` |
| Linear SVM | `LinearSVC(max_iter=2000)` |

Each model evaluated with confusion matrix heatmap, classification report, and an accuracy/F1 comparison chart. Best model saved with `joblib`.

---

## Result

All three models hit 0.88 to 0.90 weighted F1 on the Indonesian dataset. Logistic Regression was selected as the best model for both languages based on accuracy and recall on the negative class. English scores were slightly lower because of the smaller training set.

The full project produces 6 sequential notebooks, two preprocessed CSVs, and exported `.pkl` files for the best model and TF-IDF vectorizer per language.

---

## What the Analysis Reveals for Bank Jago

Negative reviews are not vague. They cluster around three recurring problems: login failures after app updates, transaction failures during QRIS payments, and slow or unhelpful customer service response. These show up consistently in both the Indonesian and English datasets, which means they are not isolated incidents.

The English-language reviews are worth separating out because that user segment tends to report different issues. They are more likely to describe problems with international payments, virtual card behavior, and Pocket isolation logic. These are not complaints that surface prominently in the Indonesian review volume, so they would get buried if both languages were analyzed together.

A model that classifies incoming reviews at scale means the product team does not need to read through 75,000 reviews manually. Filtering predicted negative reviews by keyword frequency turns raw complaints into a ranked list of what to fix first, sorted by how many users are actually affected.

---

## Project Structure

Tugas_1/
csv/
jago_reviews_raw.csv
jago_preprocessing_id.csv
jago_preprocessing_en.csv
jupyter/
1-data_scrapping.ipynb
2-Preprocessing.ipynb
3-EDA.ipynb
3-EDA_EN.ipynb
4-Sentiment_Analysis.ipynb
4-Sentiment_Analysis_EN.ipynb
README.md