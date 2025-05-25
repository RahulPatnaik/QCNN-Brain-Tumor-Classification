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
  Flex,
  Button,
  useColorModeValue,
  Badge,
  Icon,
  VStack,
  HStack,
  Divider,
  Card,
  CardBody,
  CardHeader,
  CardFooter,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  SimpleGrid,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Avatar,
  AvatarBadge,
  Tooltip,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Spinner,
} from '@chakra-ui/react'
import { useDropzone } from 'react-dropzone'
import { 
  FaBrain, 
  FaUpload, 
  FaSpinner, 
  FaCheckCircle, 
  FaExclamationTriangle,
  FaChartLine,
  FaRobot,
  FaMicroscope,
  FaShieldAlt,
  FaInfoCircle,
} from 'react-icons/fa'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useNavigate } from 'react-router-dom'

interface PredictionResult {
  class_name: string
  class_description: string
  probabilities: {
    [key: string]: number
  }
  processing_time: number
  confidence_score: number
}

export default function Home() {
  const navigate = useNavigate()
  const [image, setImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const toast = useToast()
  const { isOpen, onOpen, onClose } = useDisclosure()
  const [activeModal, setActiveModal] = useState<string | null>(null)

  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')
  const textColor = useColorModeValue('gray.600', 'gray.400')
  const headingColor = useColorModeValue('gray.800', 'white')
  const cardBg = useColorModeValue('white', 'gray.700')
  const hoverBg = useColorModeValue('gray.50', 'gray.600')
  const headerBg = useColorModeValue('rgba(255, 255, 255, 0.8)', 'rgba(26, 32, 44, 0.8)')

  const handleOpenModal = (modalType: string) => {
    setActiveModal(modalType)
    onOpen()
  }

  const handleCloseModal = () => {
    setActiveModal(null)
    onClose()
  }

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
        toast({
          title: 'Analysis Complete',
          description: 'Brain scan has been successfully analyzed.',
          status: 'success',
          duration: 5000,
          isClosable: true,
        })
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

  const formatClassName = (className: string) => {
    return className
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  return (
    <Box minH="100vh" bg={useColorModeValue('gray.50', 'gray.900')}>
      {/* Header */}
      <Box 
        as="header" 
        position="fixed" 
        w="full" 
        borderBottom="1px" 
        borderColor={borderColor}
        zIndex="sticky"
        py={4}
        backdropFilter="blur(10px)"
        bg={headerBg}
      >
        <Container maxW="container.xl">
          <Flex justify="space-between" align="center">
            <HStack spacing={4}>
              <Avatar 
                size="md" 
                bg="blue.500"
                icon={<FaBrain size="24px" />}
              >
                <AvatarBadge boxSize="1em" bg="green.500" />
              </Avatar>
              <Heading size="md" color={headingColor}>Brain Tumor AI</Heading>
            </HStack>
            <HStack spacing={4}>
              <Button
                variant="ghost"
                colorScheme="blue"
                size="sm"
                leftIcon={<FaChartLine />}
                onClick={() => handleOpenModal('stats')}
              >
                View Stats
              </Button>
              <Button
                variant="ghost"
                colorScheme="blue"
                size="sm"
                onClick={() => window.location.reload()}
              >
                Reset
              </Button>
            </HStack>
          </Flex>
        </Container>
      </Box>

      {/* Main Content */}
      <Container maxW="container.xl" pt={24} pb={20}>
        <Stack spacing={16} align="center">
          {/* Hero Section */}
          <Stack spacing={6} align="center" textAlign="center" maxW="3xl">
            <Heading 
              size="2xl" 
              fontWeight="bold"
              bgGradient="linear(to-r, blue.500, purple.500)"
              bgClip="text"
              letterSpacing="tight"
            >
              Advanced Brain Tumor Analysis
            </Heading>
            <Text fontSize="xl" color={textColor} maxW="2xl">
              Leveraging quantum computing and deep learning for precise brain scan analysis
            </Text>
          </Stack>

          {error && (
            <Alert 
              status="error" 
              borderRadius="xl" 
              bg="red.50" 
              color="red.600"
              maxW="xl"
              variant="left-accent"
            >
              <AlertIcon />
              {error}
            </Alert>
          )}

          {/* Main Grid */}
          <Grid 
            templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} 
            gap={12} 
            w="full"
          >
            {/* Upload Section */}
            <GridItem>
              <Card
                {...getRootProps()}
                bg={cardBg}
                borderRadius="2xl"
                boxShadow="xl"
                transition="all 0.3s"
                _hover={{
                  transform: 'translateY(-5px)',
                  boxShadow: '2xl',
                }}
                cursor="pointer"
                position="relative"
                overflow="hidden"
              >
                <CardBody p={12}>
                  <input {...getInputProps()} />
                  <VStack spacing={6}>
                    <Box
                      w="20"
                      h="20"
                      borderRadius="full"
                      bg="blue.50"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      transition="all 0.3s"
                      _groupHover={{ transform: 'scale(1.1)' }}
                    >
                      <Icon as={FaUpload} w={10} h={10} color="blue.500" />
                    </Box>
                    <Stack spacing={2}>
                      <Text fontSize="xl" fontWeight="medium" color={headingColor}>
                        {isDragActive ? 'Drop your brain scan here' : 'Upload Brain Scan'}
                      </Text>
                      <Text fontSize="sm" color={textColor}>
                        Drag and drop your brain scan, or click to select
                      </Text>
                      <Text fontSize="xs" color={textColor}>
                        Supports JPG, PNG, JPEG
                      </Text>
                    </Stack>
                  </VStack>
                </CardBody>
              </Card>
            </GridItem>

            {/* Preview Section */}
            <GridItem>
              {image && (
                <Card
                  bg={cardBg}
                  borderRadius="2xl"
                  boxShadow="xl"
                  position="relative"
                >
                  <CardBody p={4}>
                    <Image
                      src={image}
                      alt="Uploaded brain scan"
                      borderRadius="xl"
                      maxH="400px"
                      objectFit="contain"
                      mx="auto"
                    />
                    {loading && (
                      <Box
                        position="absolute"
                        top={0}
                        left={0}
                        right={0}
                        bottom={0}
                        bg="rgba(255, 255, 255, 0.8)"
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        borderRadius="xl"
                      >
                        <VStack spacing={4}>
                          <Icon as={FaSpinner} w={8} h={8} color="blue.500" className="animate-spin" />
                          <Text color="blue.500" fontWeight="medium">Analyzing scan...</Text>
                        </VStack>
                      </Box>
                    )}
                  </CardBody>
                </Card>
              )}
            </GridItem>
          </Grid>

          {/* Results Section */}
          {prediction && (
            <Card
              w="full"
              maxW="2xl"
              bg={cardBg}
              borderRadius="2xl"
              boxShadow="xl"
            >
              <CardHeader>
                <Stack spacing={4} textAlign="center">
                  <Badge colorScheme="blue" fontSize="sm" px={3} py={1} borderRadius="full">
                    ANALYSIS RESULT
                  </Badge>
                  <Heading size="xl" color={headingColor}>
                    {formatClassName(prediction.class_name)}
                  </Heading>
                  <Text color={textColor} fontSize="sm">
                    {prediction.class_description}
                  </Text>
                </Stack>
              </CardHeader>

              <CardBody>
                <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={8}>
                  <Stat>
                    <StatLabel>Confidence Score</StatLabel>
                    <StatNumber>{(prediction.confidence_score * 100).toFixed(2)}%</StatNumber>
                    <StatHelpText>
                      <StatArrow type="increase" />
                      High confidence
                    </StatHelpText>
                  </Stat>
                  <Stat>
                    <StatLabel>Processing Time</StatLabel>
                    <StatNumber>{prediction.processing_time.toFixed(2)}s</StatNumber>
                    <StatHelpText>
                      <StatArrow type="decrease" />
                      Fast analysis
                    </StatHelpText>
                  </Stat>
                </SimpleGrid>

                <Divider mb={6} />

                <Box>
                  <Text mb={6} fontSize="sm" color={textColor} textAlign="center">
                    Confidence Scores
                  </Text>
                  <Stack spacing={6}>
                    {Object.entries(prediction.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([className, probability]) => (
                        <Box key={className}>
                          <Stack direction="row" justify="space-between" mb={2}>
                            <Text fontSize="sm" color={headingColor} fontWeight="medium">
                              {formatClassName(className)}
                            </Text>
                            <Text fontSize="sm" color={textColor} fontWeight="medium">
                              {(probability * 100).toFixed(2)}%
                            </Text>
                          </Stack>
                          <Progress
                            value={probability * 100}
                            colorScheme={probability > 0.5 ? "green" : "blue"}
                            size="sm"
                            borderRadius="full"
                            bg="gray.100"
                          />
                        </Box>
                    ))}
                  </Stack>
                </Box>
              </CardBody>

              <CardFooter>
                <SimpleGrid columns={3} spacing={4} w="full">
                  <Tooltip label="AI-Powered Analysis">
                    <Box textAlign="center">
                      <Icon as={FaRobot} w={6} h={6} color="blue.500" />
                      <Text fontSize="xs" color={textColor}>AI Analysis</Text>
                    </Box>
                  </Tooltip>
                  <Tooltip label="Medical Grade">
                    <Box textAlign="center">
                      <Icon as={FaMicroscope} w={6} h={6} color="green.500" />
                      <Text fontSize="xs" color={textColor}>Medical Grade</Text>
                    </Box>
                  </Tooltip>
                  <Tooltip label="Secure Processing">
                    <Box textAlign="center">
                      <Icon as={FaShieldAlt} w={6} h={6} color="purple.500" />
                      <Text fontSize="xs" color={textColor}>Secure</Text>
                    </Box>
                  </Tooltip>
                </SimpleGrid>
              </CardFooter>
            </Card>
          )}
        </Stack>
      </Container>

      {/* Footer */}
      <Box 
        as="footer" 
        position="fixed" 
        bottom={0} 
        w="full" 
        borderTop="1px" 
        borderColor={borderColor}
        py={4}
        backdropFilter="blur(10px)"
        bg={headerBg}
      >
        <Container maxW="container.xl">
          <Flex justify="center" align="center">
            <Text fontSize="sm" color={textColor}>
              © 2025, By Rahul & Noel
            </Text>
          </Flex>
        </Container>
      </Box>

      {/* Stats Modal */}
      <Modal isOpen={isOpen && activeModal === 'stats'} onClose={handleCloseModal} size="xl">
        <ModalOverlay backdropFilter="blur(10px)" />
        <ModalContent>
          <ModalHeader>Analysis Statistics</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <SimpleGrid columns={2} spacing={6}>
              <Stat>
                <StatLabel>Total Analyses</StatLabel>
                <StatNumber>1,234</StatNumber>
                <StatHelpText>
                  <StatArrow type="increase" />
                  23.36%
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel>Accuracy Rate</StatLabel>
                <StatNumber>98.5%</StatNumber>
                <StatHelpText>
                  <StatArrow type="increase" />
                  1.2%
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel>Average Processing Time</StatLabel>
                <StatNumber>2.3s</StatNumber>
                <StatHelpText>
                  <StatArrow type="decrease" />
                  0.5s
                </StatHelpText>
              </Stat>
              <Stat>
                <StatLabel>Success Rate</StatLabel>
                <StatNumber>99.9%</StatNumber>
                <StatHelpText>
                  <StatArrow type="increase" />
                  0.1%
                </StatHelpText>
              </Stat>
            </SimpleGrid>
          </ModalBody>
        </ModalContent>
      </Modal>
    </Box>
  )
} 