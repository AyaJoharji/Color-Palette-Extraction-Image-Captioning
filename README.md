# Color Palette Extraction & Image Captioning

## Project Overview

This project aims to create a web application that extracts a color palette from an uploaded image and generates a bilingual caption (in both English and Arabic) that describes the image. The application leverages Hugging Face pipelines for image captioning and translation, combined with a user-friendly interface built using Gradio.

### Motivation

The idea behind this project is to simplify the process of analyzing and describing images. It combines the visual aspect of color extraction, which can assist designers and artists in choosing color schemes, with descriptive captions that enhance image understanding. The inclusion of Arabic language support helps ensure accessibility for a broader audience, particularly Arabic speakers.

## Features

1. **Color Palette Extraction**:
   - Extracts up to 8 dominant colors from the uploaded image using KMeans clustering.
   - Displays a visual representation of the extracted colors alongside their hexadecimal values, making it convenient for design applications.
  
2. **Image Captioning**:
   - Generates a descriptive caption for the uploaded image using a pre-trained image-to-text model.
   - The caption aims to provide an insightful and accurate summary of the content present in the image.
  
3. **Bilingual Captioning**:
   - The caption generated in English is translated to Arabic using a pre-trained translation model.
   - This feature makes the project useful for both English and Arabic-speaking audiences.
  
4. **User-Friendly Interface**:
   - Gradio is used to create an intuitive web-based interface that allows users to easily upload images, generate captions, and extract color palettes.

## Technology Stack

- **Python**: Programming language used for building the application.
- **Hugging Face Transformers**: Provides pre-trained models for image captioning and translation.
- **Gradio**: Used to create an easy-to-use user interface.
- **Scikit-Learn**: Utilized for KMeans clustering to extract dominant colors from images.
- **Pillow**: Image processing library used to handle and manipulate images.

## Models and Pipelines Used

### 1. **Image Captioning Model**
- **Model**: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **Pipeline**: `image-to-text`
- **Description**: This model is capable of generating detailed captions for images. It was selected because of its accuracy in creating human-like captions, making it suitable for describing diverse visual content.
- **Justification**: The BLIP (Bootstrapping Language-Image Pre-training) model was chosen for its capability to generalize well across various types of images, producing accurate and descriptive captions.

### 2. **Translation Model**
- **Model**: [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
- **Pipeline**: `translation`
- **Description**: This model is used to translate text from English to Arabic. It was selected for its ability to handle multiple languages and for its robustness in translating complex sentences.
- **Justification**: The mBART model supports 50 different languages, including Arabic, and offers high-quality translations. It ensures that the generated captions are accessible to Arabic-speaking users.

## Special Measures for Arabic Language Support

The project explicitly supports Arabic by incorporating the following measures:
- The **Translation Pipeline**: The `facebook/mbart-large-50-many-to-many-mmt` model was specifically chosen for its proficiency in translating text to Arabic. This allows the project to provide bilingual captions.
- **Post-Processing for Arabic Text**: The translated text undergoes post-processing to remove repeated words and ensure the output is coherent and meaningful in Arabic. This step was added due to the complexity of Arabic grammar and to enhance the readability of the translation.

## Expected Outputs

1. **Bilingual Caption**:
   - The application provides a caption describing the content of the uploaded image in both English and Arabic.
   - Example: 
     - **English**: "A calm lake reflecting the blue sky surrounded by green trees."
     - **Arabic**: "بحيرة هادئة تعكس السماء الزرقاء محاطة بأشجار خضراء."

2. **Color Palette**:
   - A visual representation of the dominant colors extracted from the image.
   - **Hex Codes**: A list of the hex color codes for each of the extracted colors.
   - Example: `#00aaff`, `#ffaa00`, `#228b22`, etc.
  
## Hugging Face space: [Link to hugging face space](https://huggingface.co/spaces/ayajoharji/Color_PaletteExtraction_and_ImageCaptioning)
