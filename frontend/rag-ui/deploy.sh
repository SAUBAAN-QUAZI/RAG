#!/bin/bash
# Script to deploy updated frontend to Vercel

echo "Building and deploying updated frontend to Vercel..."
echo "Using API URL: $(grep NEXT_PUBLIC_API_URL .env.local | cut -d '=' -f2)"

# Build the application
echo "Running build process..."
npm run build

# Deploy to Vercel (assuming Vercel CLI is installed)
echo "Deploying to Vercel..."
vercel --prod

echo "Deployment completed. Check the Vercel dashboard for deployment status."
echo "Frontend URL: https://rag-mocha.vercel.app" 