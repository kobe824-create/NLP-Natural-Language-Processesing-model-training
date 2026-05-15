"""
Vercel Serverless Function — Sentiment Analysis API
Endpoint: POST /api/analyze
Body: { "text": "..." }
Returns: JSON with prediction, confidence, chart_data, translation, sentence_details
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import re
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_model = None

def get_model():
    global _model
    if _model is None:
        import joblib
        model_path = os.path.join(ROOT_DIR, 'model', 'nlp_model.pkl')
        _model = joblib.load(model_path)
    return _model


def batch_translate(texts):
    """Translate English texts to Kinyarwanda via deep_translator."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='rw')
        translations = []
        for i, text in enumerate(texts):
            try:
                result = translator.translate(text)
                translations.append(result if result else text)
            except Exception:
                translations.append("[Translation unavailable]")
            if i < len(texts) - 1:
                time.sleep(0.3)
        return translations
    except Exception:
        return ["[Translation unavailable]"] * len(texts)


LABEL_NAMES = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}


class handler(BaseHTTPRequestHandler):

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)
        user_text = data.get('text', '').strip()

        if not user_text:
            return self._send_json(400, {'error': 'No text provided'})

        try:
            model = get_model()
            classes = model.classes_.tolist()

            # Overall prediction
            pred = model.predict([user_text])[0]
            proba = model.predict_proba([user_text])[0]
            prediction = LABEL_NAMES.get(pred, 'Unknown')
            confidence = float(max(proba))

            # Per-class confidence
            confidence_breakdown = {}
            for cls, prob in zip(classes, proba):
                confidence_breakdown[LABEL_NAMES.get(cls, str(cls))] = round(float(prob) * 100, 1)

            # Sentence-level analysis
            sentences = [s.strip() for s in re.split(r'[.!?]+', user_text) if s.strip()]
            if not sentences:
                sentences = [user_text]

            # Batch translate
            all_texts = [user_text] + sentences
            all_translations = batch_translate(all_texts)
            translation_rw = all_translations[0]
            sentence_translations = all_translations[1:]

            chart_data = []
            sentence_details = []

            for idx, sentence in enumerate(sentences):
                s_pred = model.predict([sentence])[0]
                s_proba = model.predict_proba([sentence])[0]
                s_conf = float(max(s_proba))
                score = float(s_pred * s_conf)
                chart_data.append(round(score, 4))

                sentence_details.append({
                    'text': sentence,
                    'sentiment': LABEL_NAMES.get(s_pred, 'Unknown'),
                    'confidence': round(s_conf * 100, 1),
                    'translation': sentence_translations[idx] if idx < len(sentence_translations) else '[N/A]'
                })

            self._send_json(200, {
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'confidence_breakdown': confidence_breakdown,
                'chart_data': chart_data,
                'translation_rw': translation_rw,
                'sentence_details': sentence_details,
                'sentences_count': len(sentences)
            })

        except Exception as e:
            self._send_json(500, {'error': str(e)})
