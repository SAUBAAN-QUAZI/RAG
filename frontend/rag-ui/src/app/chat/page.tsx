'use client';

import React from 'react';
import { Box, Heading } from '@chakra-ui/react';
import Layout from '../../components/Layout';
import Chat from '../../components/Chat';

export default function ChatPage() {
  return (
    <Layout>
      <Box>
        <Heading mb={6}>Chat with Documents</Heading>
        <Chat />
      </Box>
    </Layout>
  );
} 