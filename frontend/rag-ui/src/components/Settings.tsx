'use client';

import React, { useState, useEffect } from 'react';

interface SettingsForm {
  apiUrl: string;
  modelName: string;
  temperature: number;
  topK: number;
  chunkSize: number;
  chunkOverlap: number;
}

/**
 * Settings component for configuring the RAG system
 */
const Settings: React.FC = () => {
  const [settings, setSettings] = useState<SettingsForm>({
    apiUrl: '',
    modelName: 'gpt-3.5-turbo',
    temperature: 0.7,
    topK: 4,
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  // Load settings from localStorage on initial render
  useEffect(() => {
    const savedSettings = localStorage.getItem('ragSettings');
    if (savedSettings) {
      try {
        const parsedSettings = JSON.parse(savedSettings);
        setSettings(parsedSettings);
      } catch (error) {
        console.error('Failed to parse saved settings:', error);
      }
    }
  }, []);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    setMessage(null);

    try {
      // Save settings to localStorage
      localStorage.setItem('ragSettings', JSON.stringify(settings));

      // If we had a settings API, we would call it here
      // For now, we'll just simulate a delay
      setTimeout(() => {
        setMessage({
          text: 'Settings saved successfully',
          type: 'success',
        });
        setIsSaving(false);
      }, 500);
    } catch (error) {
      console.error('Failed to save settings:', error);
      setMessage({
        text: error instanceof Error ? error.message : 'Failed to save settings',
        type: 'error',
      });
      setIsSaving(false);
    }
  };

  // Handle input change
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    isNumber = false
  ) => {
    const { name, value } = e.target;
    setSettings((prev) => ({
      ...prev,
      [name]: isNumber ? parseFloat(value) : value,
    }));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      {/* Message display */}
      {message && (
        <div
          className={`p-4 mb-6 rounded-md ${
            message.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}
        >
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* API Settings */}
        <div className="bg-white p-6 rounded-md shadow-sm border">
          <h2 className="text-lg font-medium mb-4">API Settings</h2>
          <div className="space-y-4">
            <div>
              <label htmlFor="apiUrl" className="block text-sm font-medium text-gray-700 mb-1">
                API URL
              </label>
              <input
                type="text"
                id="apiUrl"
                name="apiUrl"
                value={settings.apiUrl}
                onChange={(e) => handleChange(e)}
                placeholder="http://localhost:8000"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Leave empty to use the default API URL
              </p>
            </div>

            <div>
              <label htmlFor="modelName" className="block text-sm font-medium text-gray-700 mb-1">
                Model
              </label>
              <select
                id="modelName"
                name="modelName"
                value={settings.modelName}
                onChange={(e) => handleChange(e)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
              </select>
            </div>

            <div>
              <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 mb-1">
                Temperature: {settings.temperature}
              </label>
              <input
                type="range"
                id="temperature"
                name="temperature"
                value={settings.temperature}
                onChange={(e) => handleChange(e, true)}
                min="0"
                max="1"
                step="0.1"
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>More Precise</span>
                <span>More Creative</span>
              </div>
            </div>
          </div>
        </div>

        {/* Retrieval Settings */}
        <div className="bg-white p-6 rounded-md shadow-sm border">
          <h2 className="text-lg font-medium mb-4">Retrieval Settings</h2>
          <div className="space-y-4">
            <div>
              <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-1">
                Top K Results: {settings.topK}
              </label>
              <input
                type="range"
                id="topK"
                name="topK"
                value={settings.topK}
                onChange={(e) => handleChange(e, true)}
                min="1"
                max="10"
                step="1"
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                Number of document chunks to retrieve for each query
              </p>
            </div>

            <div>
              <label htmlFor="chunkSize" className="block text-sm font-medium text-gray-700 mb-1">
                Chunk Size: {settings.chunkSize}
              </label>
              <input
                type="range"
                id="chunkSize"
                name="chunkSize"
                value={settings.chunkSize}
                onChange={(e) => handleChange(e, true)}
                min="256"
                max="2048"
                step="128"
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                Size of document chunks in tokens (applies to new documents)
              </p>
            </div>

            <div>
              <label
                htmlFor="chunkOverlap"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Chunk Overlap: {settings.chunkOverlap}
              </label>
              <input
                type="range"
                id="chunkOverlap"
                name="chunkOverlap"
                value={settings.chunkOverlap}
                onChange={(e) => handleChange(e, true)}
                min="0"
                max="512"
                step="32"
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                Overlap between chunks in tokens (applies to new documents)
              </p>
            </div>
          </div>
        </div>

        {/* Submit button */}
        <div>
          <button
            type="submit"
            disabled={isSaving}
            className={`w-full py-2 px-4 rounded-md text-white font-medium ${
              isSaving ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Settings; 