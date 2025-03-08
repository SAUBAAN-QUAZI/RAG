# Deployment Guide for RAG System

This guide explains how to deploy the RAG system's backend to Render and the frontend to Vercel.

## Vector Database Setup (Qdrant Cloud)

For production deployment, we recommend using Qdrant Cloud for persistent vector storage.

### Setting Up Qdrant Cloud

1. **Create a Qdrant Cloud account**
   - Visit [Qdrant Cloud](https://cloud.qdrant.io/)
   - Sign up for a free account

2. **Create a new cluster**
   - From the dashboard, click on "Create Cluster"
   - Choose a name for your cluster
   - Select a region close to your users
   - Choose "Free tier" for testing (1GB storage, 100K vectors)
   - Click "Create"

3. **Get your connection details**
   - Once your cluster is created, you'll see "Connection Details"
   - Copy the "Cluster URL" (e.g., `https://xyz-123.aws.cloud.qdrant.io`)
   - Generate an API key and copy it

4. **Configure your RAG system**
   - In your `.env` file, set:
     ```
     VECTOR_DB_TYPE=qdrant
     VECTOR_DB_URL=your_cluster_url
     VECTOR_DB_API_KEY=your_api_key
     ```

5. **Test your connection**
   - Run the test script: `python test_qdrant.py`
   - Verify that the connection works

### Qdrant vs ChromaDB

The RAG system supports both ChromaDB and Qdrant:

- **ChromaDB**: Good for local development, but data is lost when Render's ephemeral storage is cleared
- **Qdrant Cloud**: Persistent storage, scales better, more suitable for production

## Backend Deployment (Render)

### Prerequisites
- [Render](https://render.com/) account
- OpenAI API key

### Steps

1. **Fork or clone the repository to your GitHub account**

2. **Connect your GitHub repository to Render**
   - Log in to your Render account
   - Go to Dashboard and click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Choose the branch you want to deploy

3. **Configure the service**
   - Name: `rag-backend` (or your preferred name)
   - Environment: `Python`
   - Region: Choose the region closest to your users
   - Branch: `main` (or your default branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - Plan: Free (or choose a paid plan if needed)

4. **Set environment variables**
   - Click on "Environment" and add the following variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `RENDER`: `true`
     - `STORAGE_ROOT`: `/tmp`
     - `VECTOR_DB_TYPE`: `qdrant`
     - `VECTOR_DB_URL`: Your Qdrant Cloud URL
     - `VECTOR_DB_API_KEY`: Your Qdrant API key
     - `ALLOW_CORS`: `true`
     - `CORS_ORIGINS`: `https://your-frontend-url.vercel.app,http://localhost:3000` (Update with your actual frontend URL)
     - `DEBUG`: `false`

5. **Deploy the service**
   - Click "Create Web Service"
   - Wait for the build and deployment to complete

6. **Note your backend URL**
   - Once deployed, note the URL (e.g., `https://rag-backend.onrender.com`)
   - You'll need this URL for your frontend deployment

### Known Limitations on Render Free Tier

1. **Ephemeral Storage**
   - Files stored in `/tmp` will persist for the lifetime of the service instance
   - They will be lost if the service restarts or spins down from inactivity
   - Using Qdrant Cloud avoids vector storage data loss

2. **Sleep after Inactivity**
   - Free tier services spin down after 15 minutes of inactivity
   - First request after spin-down will take longer to respond
   - Consider upgrading to paid plan for consistent performance

## Frontend Deployment (Vercel)

### Prerequisites
- [Vercel](https://vercel.com/) account

### Steps

1. **Navigate to the frontend directory**
   ```bash
   cd frontend/rag-ui
   ```

2. **Install Vercel CLI (optional)**
   ```bash
   npm i -g vercel
   ```

3. **Deploy via Vercel Dashboard**
   - Log in to your Vercel account
   - Click "Add New" > "Project"
   - Import your GitHub repository
   - Configure the project:
     - Framework Preset: Next.js
     - Root Directory: `frontend/rag-ui`
     - Build Command: `npm run build`
     - Output Directory: `.next`

4. **Set environment variables in Vercel**
   - Click on "Environment Variables" and add:
     - `NEXT_PUBLIC_API_URL`: Your Render backend URL (e.g., `https://rag-backend.onrender.com`)

5. **Deploy**
   - Click "Deploy"
   - Wait for the build and deployment to complete

6. **Visit your deployed frontend**
   - Once deployed, Vercel will provide you with a URL for your frontend
   - You can also configure a custom domain in the Vercel dashboard

## Using the Deployed Application

1. **Access your frontend application** using the Vercel URL
2. **Upload documents** through the frontend interface
3. **Query the documents** using natural language questions

## Troubleshooting

### Backend Issues
- **Connection Timeouts**: Increase the timeout in `config.ts` for large documents
- **Memory Errors**: Monitor memory usage on Render and consider upgrading the plan
- **Cold Starts**: Initial requests after inactivity may time out; reload and try again
- **Qdrant Connection**: Verify your Qdrant URL and API key if vector search fails

### Frontend Issues
- **CORS Errors**: Ensure the backend's `CORS_ORIGINS` includes your frontend URL
- **Upload Failures**: Check file size limits and network connectivity
- **Slow Responses**: Large documents take longer to process; be patient

## Upgrading to Production

For a more robust production deployment, consider these improvements:

1. **Use Cloud Storage** (S3, Firebase, etc.) for document persistence
2. **Upgrade Qdrant Plan** for higher storage limits and better performance
3. **Add Authentication** to protect your API endpoints
4. **Use a CDN** for faster frontend content delivery
5. **Set up Monitoring** to track usage and detect issues 