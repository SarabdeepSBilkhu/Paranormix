# Deployment Guide: GitHub Pages + Railway Backend

Paranormix uses a hybrid deployment model. Since the backend requires a Python environment and an ML model, it cannot be hosted directly on GitHub Pages. Instead, we host the frontend on GitHub and the backend on a specialized provider.

## Step 1: Deploy the Backend to Railway

1. **Create a Railway Account**: Sign up at [Railway.app](https://railway.app).
2. **Create New Project**: Click "New Project" and select "Deploy from GitHub repo".
3. **Select Repository**: Choose your Paranormix repository.
4. **Configure Build/Start**:
   Railway should automatically detect the `requirements.txt`. If prompted for a start command, use:
   ```bash
   uvicorn src.backend.main:app --host 0.0.0.0 --port $PORT
   ```
5. **Add Variables**: Go to the "Variables" tab in your Railway service and add:
   - `GROQ_API_KEY`: [Your Groq Key]
6. **Deploy**: Railway will deploy the service and provide a public URL (e.g., `https://paranormix-production.up.railway.app`).

## Step 2: Deploy the Frontend (GitHub Pages)

1. **Push your code** to a GitHub repository.
2. **Settings**:
   - Go to **Settings > Pages**.
   - Under "Build and deployment", select **Deploy from a branch**.
   - Select your main branch and the `/docs` folder.
3. **Wait for deployment**: Your site will be live at `https://[username].github.io/[repo]`.

## Step 3: Connect Frontend to Backend

Once your backend is live (e.g., `https://paranormix-production.up.railway.app`), you need to tell the frontend where it is.

1. Open your live GitHub Pages URL.
2. Open the browser console (F12).
3. Run this command:
   ```javascript
   localStorage.setItem("paranormix_api_url", "https://your-backend-url.com");
   ```
4. Refresh the page. Paranormix will now communicate with your live server!

---

## Technical Maintenance

- **Model Updates**: To update the model, retrain locally and push the new `models/ghost_model.pkl` to GitHub. Your backend provider will automatically redeploy.
- **Session Memory**: Remember that sessions are stored in-memory. If your backend server restarts (common on free tiers), active investigations will be reset.
