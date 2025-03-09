import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    message: 'RAG UI frontend is running',
    time: new Date().toISOString(),
  });
} 