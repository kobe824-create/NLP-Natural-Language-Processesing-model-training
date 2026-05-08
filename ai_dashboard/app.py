from flask import Flask, render_template, request, flash
import joblib
import re
import time
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = 'ai-dashboard-secret-key-2026'

# Reusable translator instance
translator = GoogleTranslator(source='en', target='rw')


def batch_translate_kinyarwanda(texts: list) -> list:
    """
    Translate a list of English strings to Kinyarwanda in one batch.
    Uses a single GoogleTranslator instance and translates each text
    with a small delay to avoid rate-limiting.
    Returns a list of translated strings (same order as input).
    """
    translations = []
    for i, text in enumerate(texts):
        try:
            result = translator.translate(text)
            translations.append(result if result else text)
        except Exception as e:
            app.logger.warning(f"Translation failed for: {text[:50]}... Error: {e}")
            translations.append("[Translation unavailable]")
        # Small delay between calls to prevent rate-limiting
        if i < len(texts) - 1:
            time.sleep(0.5)
    return translations


@app.route('/', methods=['GET', 'POST'])
def expert_portal():
    analyzed = False
    prediction = None
    confidence = 0.0
    chart_data = []
    user_text = ""
    confidence_breakdown = {}   # per-class confidence scores
    translation_rw = ""         # full Kinyarwanda translation
    sentence_details = []       # per-sentence: text, sentiment, confidence, translation

    if request.method == 'POST':
        user_text = request.form.get('text_input', '')

        if user_text.strip():
            try:
                # Load the NLP pipeline
                model = joblib.load('model/nlp_model.pkl')
                classes = model.classes_.tolist()

                # ── Overall prediction ───────────────────────────────
                pred = model.predict([user_text])[0]
                proba = model.predict_proba([user_text])[0]

                label_names = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
                prediction = label_names.get(pred, 'Unknown')
                confidence = float(max(proba))

                # Per-class confidence breakdown
                for cls, prob in zip(classes, proba):
                    confidence_breakdown[label_names.get(cls, str(cls))] = round(float(prob) * 100, 1)

                # ── Sentence-by-sentence trajectory ──────────────────
                sentences = [s.strip() for s in re.split(r'[.!?]+', user_text) if s.strip()]
                if not sentences:
                    sentences = [user_text]

                # ── Batch translate: full text + all sentences ────────
                all_texts_to_translate = [user_text] + sentences
                all_translations = batch_translate_kinyarwanda(all_texts_to_translate)
                translation_rw = all_translations[0]
                sentence_translations = all_translations[1:]

                for idx, sentence in enumerate(sentences):
                    s_pred = model.predict([sentence])[0]
                    s_proba = model.predict_proba([sentence])[0]
                    s_conf = float(max(s_proba))
                    score = s_pred * s_conf
                    chart_data.append(score)

                    sentence_details.append({
                        'text': sentence,
                        'sentiment': label_names.get(s_pred, 'Unknown'),
                        'confidence': round(s_conf * 100, 1),
                        'translation': sentence_translations[idx],
                    })

                analyzed = True

            except Exception as e:
                flash(f"Error loading NLP model: {e}. Please run train_nlp_model.py first.", "error")

    return render_template('expert_portal.html',
                           analyzed=analyzed,
                           user_text=user_text,
                           prediction=prediction,
                           confidence=confidence,
                           chart_data=chart_data,
                           confidence_breakdown=confidence_breakdown,
                           translation_rw=translation_rw,
                           sentence_details=sentence_details)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
