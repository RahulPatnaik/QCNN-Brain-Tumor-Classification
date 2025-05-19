# Brain Tumor Classification Frontend

This is the frontend application for the Quantum Hybrid ResNet Brain Tumor Classification system. It provides an intuitive user interface for uploading brain scan images and viewing classification results.

## Prerequisites

- Node.js (v18 or higher)
- npm (v9 or higher)
- Python 3.10+ (for backend)

## Setup Instructions

1. Install frontend dependencies:
```bash
cd Frontend
npm install
```

2. Install backend dependencies:
```bash
cd ../Backend
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server (in one terminal):
```bash
cd Backend
uvicorn main:app --reload
```

2. Start the frontend development server (in another terminal):
```bash
cd Frontend
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Features

- Drag and drop interface for image upload
- Real-time image preview
- Detailed classification results with confidence scores
- Modern and responsive design
- Error handling and loading states

## Technologies Used

- React with TypeScript
- Vite
- Chakra UI
- React Dropzone
- Axios
- FastAPI (Backend)
- PyTorch (Backend)
- PennyLane (Backend)

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config({
  extends: [
    // Remove ...tseslint.configs.recommended and replace with this
    ...tseslint.configs.recommendedTypeChecked,
    // Alternatively, use this for stricter rules
    ...tseslint.configs.strictTypeChecked,
    // Optionally, add this for stylistic rules
    ...tseslint.configs.stylisticTypeChecked,
  ],
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
```
