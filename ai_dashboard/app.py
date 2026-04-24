from flask import Flask, render_template, request, flash
import joblib
import re

app = Flask(__name__)
app.secret_key = 'ai-dashboard-secret-key-2026'

@app.route('/', methods=['GET', 'POST'])
def expert_portal():
    analyzed = False
    prediction = None
    confidence = 0.0
    chart_data = []
    user_text = ""
    
    if request.method == 'POST':
        user_text = request.form.get('text_input', '')
        
        if user_text.strip():
            try:
                # Load the NLP pipeline
                model = joblib.load('model/nlp_model.pkl')
                
                # Predict overall sentiment
                pred = model.predict([user_text])[0]
                proba = model.predict_proba([user_text])[0]
                
                if pred == 1:
                    prediction = 'Positive'
                elif pred == -1:
                    prediction = 'Negative'
                else:
                    prediction = 'Neutral'
                    
                confidence = max(proba)
                
                # Generate sentence-by-sentence trajectory
                # Simple sentence split by punctuation
                sentences = [s.strip() for s in re.split(r'[.!?]+', user_text) if s.strip()]
                
                if not sentences:
                    sentences = [user_text]
                    
                for sentence in sentences:
                    s_pred = model.predict([sentence])[0]
                    # Score mapping: 1 -> 1.0, 0 -> 0.0, -1 -> -1.0
                    # Modulate by confidence to give it smooth variance
                    s_proba = max(model.predict_proba([sentence])[0])
                    score = s_pred * s_proba
                    chart_data.append(score)
                    
                analyzed = True
                
            except Exception as e:
                flash(f"Error loading NLP model: {e}. Please run train_nlp_model.py first.", "error")

    return render_template('expert_portal.html',
                           analyzed=analyzed,
                           user_text=user_text,
                           prediction=prediction,
                           confidence=confidence,
                           chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
