#!/bin/bash
# Script to deploy updated frontend to Vercel with clean build

echo "Building and deploying updated frontend to Vercel..."
echo "Using API URL: $(grep NEXT_PUBLIC_API_URL .env.local | cut -d '=' -f2)"
echo "Current build version: $(grep NEXT_PUBLIC_VERSION .env.local | cut -d '=' -f2)"

# Clean existing build artifacts
echo "Cleaning previous builds..."
rm -rf .next
rm -rf node_modules/.cache

# Clean install dependencies
echo "Reinstalling dependencies..."
npm ci

# Build the application
echo "Running clean build process..."
NEXT_PUBLIC_API_URL=$(grep NEXT_PUBLIC_API_URL .env.local | cut -d '=' -f2) npm run build

# Deploy to Vercel (assuming Vercel CLI is installed)
echo "Deploying to Vercel..."
vercel --prod

echo "Deployment completed. Check the Vercel dashboard for deployment status."
echo "Frontend URL: https://rag-mocha.vercel.app"
echo "After deployment, please verify that the API URL is correctly set by checking the browser console." 