import numpy as np

def explain_prediction(text, model, vectorizer, top_n=5):

    X = vectorizer.transform([text])


    prediction = model.predict(X)[0]


    prob = model.predict_proba(X)[0]
    risk_score = round(max(prob) * 100, 2)


    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        weights = model.coef_[0]
    else:
        weights = model.feature_log_prob_[1]


    word_scores = X.toarray()[0] * weights
    top_indices = np.argsort(word_scores)[-top_n:][::-1]

    keywords = [feature_names[i] for i in top_indices if word_scores[i] > 0]

    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "risk_percentage": risk_score,
        "keywords": keywords
    }

