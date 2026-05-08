"""Quick end-to-end test for the sentiment dashboard."""
import requests
import re

print("=== Testing Sentiment Dashboard ===\n")

r = requests.post("http://127.0.0.1:5001", data={
    "text_input": "I absolutely love this product, it changed my life! "
                  "However, the delivery was terrible and took way too long. "
                  "Overall, it is okay I guess."
})
html = r.text
print("Status:", r.status_code)

# 1. Sentiment
for s in ["Positive", "Negative", "Neutral"]:
    if s in html:
        print(f"[OK] Sentiment detected: {s}")

# 2. Confidence breakdown
breakdown = re.findall(
    r'conf-label[^>]*>\s*(\w+)\s*</span>.*?conf-pct[^>]*>(\d+\.\d+)%',
    html, re.DOTALL
)
if breakdown:
    print("[OK] Confidence breakdown:")
    for name, pct in breakdown:
        print(f"     {name}: {pct}%")
else:
    print("[FAIL] No confidence breakdown found")

# 3. Kinyarwanda translation
if "Ikinyarwanda" in html:
    trans_match = re.search(r'translation-text[^>]*>\s*([^<]+)', html)
    if trans_match:
        t = trans_match.group(1).strip()
        if t and "unavailable" not in t.lower():
            print(f"[OK] Kinyarwanda translation: {t}")
        else:
            print(f"[WARN] Translation: {t}")
    else:
        print("[WARN] Translation panel found but no text")
else:
    print("[FAIL] Kinyarwanda panel missing")

# 4. Sentence detail table
rows = re.findall(
    r'<tr>\s*<td[^>]*>\d+</td>\s*<td>([^<]+)</td>\s*<td>\s*<span class="badge badge-(\w+)">',
    html
)
if rows:
    print(f"[OK] Sentence table: {len(rows)} sentences")
    for text, sentiment in rows:
        print(f"     \"{text.strip()}\" -> {sentiment}")
else:
    print("[FAIL] No sentence table rows found")

# 5. Per-sentence translations
trans_cells = re.findall(r'td-translation[^>]*>([^<]+)<', html)
real_trans = [t.strip() for t in trans_cells if t.strip()]
if real_trans:
    print(f"[OK] Sentence translations: {len(real_trans)}")
    for t in real_trans:
        print(f"     -> {t}")
else:
    print("[FAIL] No sentence translations found")

# 6. Chart data
chart_match = re.search(r'const chartData = (\[[^\]]+\])', html)
if chart_match:
    print(f"[OK] Chart data: {chart_match.group(1)}")
else:
    print("[FAIL] No chart data")

print("\n=== Test Complete ===")
