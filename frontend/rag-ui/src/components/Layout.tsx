'use client';

import React, { ReactNode } from 'react';
import { Box, Flex, Container, Heading, Text, HStack, Link as ChakraLink, useColorModeValue } from '@chakra-ui/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface LayoutProps {
  children: ReactNode;
}

/**
 * Main layout component for the RAG UI
 */
const Layout: React.FC<LayoutProps> = ({ children }) => {
  const pathname = usePathname();
  
  // Navigation items
  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Upload Documents', path: '/upload' },
    { name: 'Chat', path: '/chat' }
  ];
  
  return (
    <Flex direction="column" minH="100vh">
      {/* Header */}
      <Box as="header" bg="primary.600" color="white" boxShadow="md">
        <Container maxW="container.xl" py={3}>
          <Flex justify="space-between" align="center">
            <Heading size="md">RAG UI</Heading>
            <HStack spacing={6} as="nav">
              {navItems.map((item) => (
                <ChakraLink
                  key={item.path}
                  as={Link}
                  href={item.path}
                  fontWeight={pathname === item.path ? "bold" : "normal"}
                  borderBottom={pathname === item.path ? "2px" : "0"}
                  _hover={{ textDecoration: 'none', opacity: 0.8 }}
                >
                  {item.name}
                </ChakraLink>
              ))}
            </HStack>
          </Flex>
        </Container>
      </Box>
      
      {/* Main content */}
      <Box flex="1" bg={useColorModeValue('gray.50', 'gray.900')}>
        <Container maxW="container.xl" py={8}>
          {children}
        </Container>
      </Box>
      
      {/* Footer */}
      <Box as="footer" bg="gray.100" borderTop="1px" borderColor="gray.200">
        <Container maxW="container.xl" py={4}>
          <Text textAlign="center" fontSize="sm" color="gray.600">
            RAG UI &copy; {new Date().getFullYear()} - Powered by Next.js and FastAPI
          </Text>
        </Container>
      </Box>
    </Flex>
  );
};

export default Layout; 