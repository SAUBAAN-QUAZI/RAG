#!/bin/bash
# Script to deploy updated frontend to Vercel with clean build

echo "Building and deploying updated frontend to Vercel..."
echo "Using API URL: $(grep NEXT_PUBLIC_API_URL .env.local | cut -d '=' -f2)"
echo "Current build version: $(grep NEXT_PUBLIC_VERSION .env.local | cut -d '=' -f2)"

# Update Vercel environment variables directly
echo "Updating Vercel environment variables..."
API_URL=$(grep NEXT_PUBLIC_API_URL .env.local | cut -d '=' -f2)

# Update environment variables directly in Vercel
if command -v vercel &> /dev/null; then
    echo "Setting NEXT_PUBLIC_API_URL in Vercel..."
    vercel env add NEXT_PUBLIC_API_URL production <<< "$API_URL"
    
    echo "Setting NEXT_PUBLIC_VERSION in Vercel..."
    VERSION=$(grep NEXT_PUBLIC_VERSION .env.local | cut -d '=' -f2)
    vercel env add NEXT_PUBLIC_VERSION production <<< "$VERSION"
    
    echo "Environment variables updated in Vercel."
else
    echo "Vercel CLI not found. Please manually set environment variables in the Vercel dashboard:"
    echo "NEXT_PUBLIC_API_URL=$API_URL"
    echo "NEXT_PUBLIC_VERSION=$(grep NEXT_PUBLIC_VERSION .env.local | cut -d '=' -f2)"
fi

# Clean existing build artifacts
echo "Cleaning previous builds..."
rm -rf .next
rm -rf node_modules/.cache

# Clean install dependencies
echo "Reinstalling dependencies..."
npm ci

# Build the application
echo "Running clean build process..."
NEXT_PUBLIC_API_URL="$API_URL" npm run build

# Deploy to Vercel (assuming Vercel CLI is installed)
echo "Deploying to Vercel..."
vercel --prod --build-env NEXT_PUBLIC_API_URL="$API_URL"

echo "Deployment completed. Check the Vercel dashboard for deployment status."
echo "Frontend URL: https://rag-mocha.vercel.app"
echo "After deployment, please verify that the API URL is correctly set by checking the browser console." 