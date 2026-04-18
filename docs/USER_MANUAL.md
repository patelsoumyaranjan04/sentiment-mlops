# User Manual
## SentiAI — Amazon Review Sentiment Analyzer
**Version:** 1.0.0

---

## What Does This App Do?

SentiAI reads Amazon product review text and tells you instantly whether the review is **positive** (the customer liked the product) or **negative** (the customer was unhappy). It uses a deep learning model trained on thousands of real Amazon reviews.

---

## Getting Started

### Step 1 — Open the App
Open your web browser and go to:
```
http://localhost:3000
```
You should see the SentiAI home screen with a dark background and a text box.

> **Note:** The app must be running on your computer first. If you see a blank screen or an error, contact your system administrator.

---

## Page 1: Analyze Reviews

This is the main page where you analyze review text.

### Analyzing a Single Review

1. Click inside the large text box labeled **"Paste or type an Amazon product review here..."**
2. Type or paste the review text you want to analyze.
3. Click the purple **"✨ Analyze Sentiment"** button.
4. Wait 1–2 seconds. A result card will appear below the button.

**Reading the result:**
- **😊 POSITIVE** — the review expresses satisfaction with the product
- **😞 NEGATIVE** — the review expresses dissatisfaction
- The **percentage** (e.g., "94% confidence") shows how certain the model is
- The **colored bar** shows the split between positive and negative probability
- **Inference latency** shows how fast the analysis was (in milliseconds)

**Using example reviews:**
Click any of the **Example 1, 2, 3...** buttons at the top to automatically fill the text box with a sample review. This is a quick way to try the app.

---

### Analyzing Multiple Reviews at Once (Batch Mode)

1. Click the **"Batch Analysis"** tab at the top of the page.
2. Type or paste multiple reviews — **one review per line**.
3. You can analyze up to **50 reviews** at a time.
4. Click the **"✨ Analyze N Reviews"** button.
5. A summary appears showing how many reviews were positive and negative, with a positive rate percentage.
6. Scroll down to see each review's individual result.

---

### Recent Predictions Panel (Right Side)

The panel on the right side keeps a history of your last 20 predictions during your session.
- Each item shows the start of the review text, the sentiment emoji, confidence %, and the time.
- Click **"Clear"** to reset the history.
- The history resets if you reload the page.

---

## Page 2: ML Pipeline

Click the **"⚙️ ML Pipeline"** button in the top navigation bar to view the pipeline page.

### API Status Section
This shows whether the backend services are running:
- **🟢 UP** — the service is running normally
- **🔴 DOWN** — the service is not reachable
- Click **"↻ Refresh"** to check the current status.
- Blue links (e.g., "localhost:5000 ↗") open the tool's own interface in a new tab.

### Pipeline Stages
A visual walkthrough of all 7 stages of the ML system — from data collection to live serving. Each stage shows what tool it uses and what it produces.

### Tech Stack
A quick reference table showing all technologies used in the project.

---

## Troubleshooting

| Problem | What to Try |
|---|---|
| "⚠️ Failed to fetch" error | The backend API is not running. Start it with `docker compose up`. |
| The result says "Model not ready" | The model is still loading. Wait 20–30 seconds and try again. |
| Very low confidence (close to 50%) | The review text may be ambiguous or too short. Try a longer, clearer review. |
| The page looks broken / unstyled | Try a hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac). |
| App not opening at all | Make sure Docker is running and you used `docker compose up --build`. |

---

## Tips for Best Results

- Use **complete sentences** — the model understands context better than single words.
- Very short inputs like "good" or "bad" will produce lower-confidence results.
- The model was trained on English text — non-English reviews may give inaccurate results.
- Technical jargon or very niche product descriptions may reduce accuracy.

---

## Privacy

All review text you enter is processed **locally on your machine** and is never sent to any external server or stored permanently. Your prediction history exists only in your browser tab and disappears when you close or refresh the page.
