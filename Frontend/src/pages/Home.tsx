import { useState } from 'react'
import {
  Box,
  Container,
  Heading,
  Stack,
  Text,
  useToast,
  Progress,
  Image,
  Grid,
  GridItem,
  Alert,
  AlertIcon,
} from '@chakra-ui/react'
import { useDropzone } from 'react-dropzone'
import { FaBrain } from 'react-icons/fa'
import axios from 'axios'

interface PredictionResult {
  class_name: string
  probabilities: {
    [key: string]: number
  }
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const toast = useToast()

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setError(null)
    setPrediction(null)

    const reader = new FileReader()
    reader.onload = () => {
      setImage(reader.result as string)
    }
    reader.readAsDataURL(file)

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    try {
      const response = await axios.post('http://localhost:8000/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      
      if (response.data && response.data.class_name) {
        setPrediction(response.data)
      } else {
        throw new Error('Invalid response format from server')
      }
    } catch (error: any) {
      console.error('Prediction error:', error)
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to get prediction'
      setError(errorMessage)
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setLoading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
    maxFiles: 1,
  })

  return (
    <Box 
      minH="100vh" 
      bg="gray.50"
      backgroundImage="linear-gradient(to bottom, #ffffff, #f8f9fa)"
    >
      <Container maxW="container.xl" py={20}>
        <Stack spacing={16} align="center">
          {/* Header Section */}
          <Stack spacing={4} align="center" textAlign="center">
            <Heading 
              size="2xl" 
              fontWeight="bold"
              bgGradient="linear(to-r, blue.600, purple.600)"
              bgClip="text"
              transition="all 0.3s"
              _hover={{ transform: 'scale(1.02)' }}
            >
              <FaBrain style={{ display: 'inline', marginRight: '10px' }} />
              Brain Tumor Classification
            </Heading>
            <Text fontSize="xl" color="gray.600" maxW="2xl">
              Advanced AI-powered analysis of brain scans using quantum computing technology
            </Text>
          </Stack>

          {error && (
            <Alert 
              status="error" 
              borderRadius="xl" 
              bg="red.50" 
              color="red.600"
              maxW="xl"
              transition="all 0.3s"
            >
              <AlertIcon />
              {error}
            </Alert>
          )}

          {/* Main Content */}
          <Grid 
            templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} 
            gap={12} 
            w="full"
            transition="all 0.3s"
          >
            {/* Upload Section */}
            <GridItem>
              <Box
                {...getRootProps()}
                p={12}
                border="2px dashed"
                borderColor={isDragActive ? 'blue.500' : 'gray.200'}
                borderRadius="2xl"
                textAlign="center"
                cursor="pointer"
                bg="white"
                boxShadow="xl"
                transition="all 0.3s"
                _hover={{
                  borderColor: 'blue.500',
                  transform: 'translateY(-5px)',
                  boxShadow: '2xl',
                }}
                animation={isDragActive ? 'pulse 1s infinite' : 'none'}
                sx={{
                  '@keyframes pulse': {
                    '0%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.05)' },
                    '100%': { transform: 'scale(1)' },
                  },
                }}
              >
                <input {...getInputProps()} />
                <Stack spacing={6}>
                  <Box
                    w="16"
                    h="16"
                    mx="auto"
                    borderRadius="full"
                    bg="blue.50"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                  >
                    <FaBrain size="32px" color="#3182CE" />
                  </Box>
                  <Text fontSize="xl" fontWeight="medium" color="gray.700">
                    {isDragActive
                      ? 'Drop your brain scan here'
                      : 'Drag and drop your brain scan, or click to select'}
                  </Text>
                  <Text fontSize="sm" color="gray.500">
                    Supports JPG, PNG, JPEG
                  </Text>
                </Stack>
              </Box>
            </GridItem>

            {/* Preview Section */}
            <GridItem>
              {image && (
                <Box
                  p={4}
                  bg="white"
                  borderRadius="2xl"
                  boxShadow="xl"
                  transition="all 0.3s"
                >
                  <Image
                    src={image}
                    alt="Uploaded brain scan"
                    borderRadius="xl"
                    maxH="400px"
                    objectFit="contain"
                    mx="auto"
                  />
                </Box>
              )}
            </GridItem>
          </Grid>

          {/* Loading Indicator */}
          {loading && (
            <Box w="full" maxW="xl" transition="all 0.3s">
              <Text mb={2} textAlign="center" color="gray.600">
                Analyzing scan...
              </Text>
              <Progress 
                size="xs" 
                isIndeterminate 
                colorScheme="blue"
                borderRadius="full"
                bg="blue.50"
              />
            </Box>
          )}

          {/* Results Section */}
          {prediction && (
            <Box
              w="full"
              maxW="2xl"
              p={8}
              bg="white"
              borderRadius="2xl"
              boxShadow="xl"
              transition="all 0.3s"
            >
              <Stack spacing={6}>
                <Stack spacing={2} textAlign="center">
                  <Text fontSize="sm" color="blue.600" fontWeight="medium">
                    ANALYSIS RESULT
                  </Text>
                  <Heading size="xl" color="gray.800">
                    {prediction.class_name}
                  </Heading>
                </Stack>

                <Box>
                  <Text mb={4} fontSize="sm" color="gray.600" textAlign="center">
                    Confidence Scores
                  </Text>
                  <Stack spacing={4}>
                    {Object.entries(prediction.probabilities).map(([className, probability]) => (
                      <Box key={className}>
                        <Stack direction="row" justify="space-between" mb={1}>
                          <Text fontSize="sm" color="gray.700">
                            {className}
                          </Text>
                          <Text fontSize="sm" color="gray.600" fontWeight="medium">
                            {(probability * 100).toFixed(2)}%
                          </Text>
                        </Stack>
                        <Progress
                          value={probability * 100}
                          colorScheme="blue"
                          size="sm"
                          borderRadius="full"
                          bg="blue.50"
                        />
                      </Box>
                    ))}
                  </Stack>
                </Box>
              </Stack>
            </Box>
          )}
        </Stack>
      </Container>
    </Box>
  )
} 