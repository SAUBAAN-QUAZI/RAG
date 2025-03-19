'use client';

import React from 'react';
import { Box, Heading, Text, Button, SimpleGrid, Icon, VStack, Card, CardBody, CardHeader } from '@chakra-ui/react';
import { FaUpload, FaComments, FaDatabase, FaRobot } from 'react-icons/fa';
import Link from 'next/link';
import Layout from '../components/Layout';

export default function Home() {
  return (
    <Layout>
      <VStack spacing={10} align="stretch">
        {/* Hero Section */}
        <Box 
          p={10} 
          borderRadius="lg" 
          bg="primary.600" 
          color="white"
          boxShadow="xl"
        >
          <Heading size="2xl" mb={4}>Retrieval-Augmented Generation</Heading>
          <Text fontSize="xl" mb={6}>
            Upload documents and chat with your data using AI-powered question answering.
          </Text>
          <Button 
            as={Link} 
            href="/upload" 
            size="lg" 
            colorScheme="blue" 
            bg="white" 
            color="primary.600"
            _hover={{ bg: 'gray.100' }}
          >
            Get Started
          </Button>
        </Box>
        
        {/* Features Section */}
        <Box>
          <Heading size="xl" mb={8} textAlign="center">Features</Heading>
          <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={8}>
            <FeatureCard 
              icon={FaUpload} 
              title="Document Upload" 
              description="Upload PDF documents to train the RAG system"
            />
            <FeatureCard 
              icon={FaDatabase} 
              title="Vector Storage" 
              description="Documents are processed and stored as vector embeddings"
            />
            <FeatureCard 
              icon={FaRobot} 
              title="AI Generation" 
              description="LLM-powered responses based on your documents"
            />
            <FeatureCard 
              icon={FaComments} 
              title="Interactive Chat" 
              description="Chat interface for natural conversation with your data"
            />
          </SimpleGrid>
        </Box>
      </VStack>
    </Layout>
  );
}

interface FeatureCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description }) => {
  return (
    <Card shadow="md" transition="all 0.3s" _hover={{ transform: 'translateY(-5px)' }}>
      <CardHeader>
        <Icon as={icon} boxSize={10} color="primary.500" mb={2} />
        <Heading size="md">{title}</Heading>
      </CardHeader>
      <CardBody>
        <Text>{description}</Text>
      </CardBody>
    </Card>
  );
};
