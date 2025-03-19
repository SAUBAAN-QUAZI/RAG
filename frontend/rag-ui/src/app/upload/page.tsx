'use client';

import React from 'react';
import { Box, Heading } from '@chakra-ui/react';
import Layout from '../../components/Layout';
import DocumentUpload from '../../components/DocumentUpload';

export default function UploadPage() {
  return (
    <Layout>
      <Box>
        <Heading mb={6}>Upload Documents</Heading>
        <DocumentUpload />
      </Box>
    </Layout>
  );
} 