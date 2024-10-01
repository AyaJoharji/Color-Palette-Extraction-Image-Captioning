# app.py

# Import Libraries
import numpy as np
import gradio as gr
from sklearn.cluster import KMeans
from transformers import pipeline
from PIL import Image, ImageDraw
import requests
from io import BytesIO

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load pipelines globally to avoid reloading on each inference
print("Loading pipelines...")

# Image Captioning Pipeline
# Using Salesforce/blip-image-captioning-base for generating image captions
caption_pipeline = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# Translation Pipeline
# Using facebook/mbart-large-50-many-to-many-mmt for translations
# This model supports multiple languages and provides better translation quality for Arabic
translation_pipeline = pipeline(
    "translation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    tokenizer="facebook/mbart-large-50-many-to-many-mmt",
    src_lang="en_XX",
    tgt_lang="ar_AR"
)

print("Pipelines loaded successfully.")

# Define a list of image URLs for examples
image_examples = [
    ["https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=512"],
    ["https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=512"],
    ["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=512"],
    ["https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=512"],
    ["https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=512"],
    ["https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?w=512"],
    ["https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=512"],
    ["https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=512"],
]

# Function to Load and Process Image
def load_image(image):
    """
    Converts the input image to a numpy array and resizes it.
    
    Args:
        image (PIL.Image.Image): The input image.
    
    Returns:
        resized_image_np (numpy.ndarray): The resized image as a numpy array.
    """
    # Convert PIL image to numpy array (RGB)
    image_np = np.array(image.convert('RGB'))
    
    # Resize the image to (300, 300) for consistent processing
    resized_image = image.resize((300, 300), resample=Image.LANCZOS)
    resized_image_np = np.array(resized_image)
    
    return resized_image_np

# Function to Extract Dominant Colors from the Image
def extract_colors(image, k=8):
    """
    Uses KMeans clustering to extract dominant colors from the image.
    
    Args:
        image (numpy.ndarray): The input image as a numpy array.
        k (int): The number of clusters (colors) to extract.
    
    Returns:
        colors (numpy.ndarray): An array of the dominant colors.
    """
    # Flatten the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Normalize pixel values to [0, 1]
    pixels = pixels / 255.0
    pixels = pixels.astype(np.float64)
    
    # Apply KMeans clustering to find dominant colors
    kmeans = KMeans(
        n_clusters=k,
        random_state=0,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(pixels)
    
    # Convert normalized colors back to 0-255 scale
    colors = (kmeans.cluster_centers_ * 255).astype(int)
    return colors

# Function to Create an Image for the Color Palette
def create_palette_image(colors):
    """
    Creates a visual representation of the color palette.
    
    Args:
        colors (numpy.ndarray): An array of the dominant colors.
    
    Returns:
        palette_image (PIL.Image.Image): The generated color palette image.
    """
    num_colors = len(colors)
    palette_height = 100
    palette_width = 100 * num_colors
    palette_image = Image.new(
        "RGB",
        (palette_width, palette_height)
    )
    
    draw = ImageDraw.Draw(palette_image)
    for i, color in enumerate(colors):
        # Ensure color values are within valid range and integers
        color = tuple(np.clip(color, 0, 255).astype(int))
        # Draw rectangles for each color
        draw.rectangle(
            [i * 100, 0, (i + 1) * 100, palette_height],
            fill=color
        )
    
    return palette_image

# Function to Display Color Palette as Hex Codes
def display_palette(colors):
    """
    Converts RGB colors to hexadecimal format.
    
    Args:
        colors (numpy.ndarray): An array of the dominant colors.
    
    Returns:
        hex_colors (list): A list of hex color codes.
    """
    hex_colors = []
    for color in colors:
        # Ensure color values are within valid range and integers
        color = np.clip(color, 0, 255).astype(int)
        # Convert to hex code
        hex_color = "#{:02x}{:02x}{:02x}".format(
            color[0],
            color[1],
            color[2]
        )
        hex_colors.append(hex_color)
    return hex_colors

# Function to Generate Image Caption Using Pipeline
def generate_caption(image):
    """
    Generates a caption for the input image using a pre-trained model.
    
    Args:
        image (PIL.Image.Image): The input image.
    
    Returns:
        caption (str): The generated caption.
    """
    # Use the captioning pipeline to generate a caption
    result = caption_pipeline(image)
    caption = result[0]['generated_text']
    return caption

# Function to Translate Caption to Arabic Using Pipeline
def translate_to_arabic(text):
    """
    Translates English text to Arabic using a pre-trained model with enhanced post-processing.
    
    Args:
        text (str): The English text to translate.
    
    Returns:
        translated_text (str): The translated Arabic text.
    """
    try:
        # Use the translation pipeline to translate the text
        result = translation_pipeline(text)
        translated_text = result[0]['translation_text']
        
        # Post-processing to remove repeated words
        words = translated_text.split()
        seen = set()
        cleaned_words = []
        previous_word = ""
        for word in words:
            if word != previous_word:
                cleaned_words.append(word)
                seen.add(word)
            previous_word = word
        cleaned_translated_text = ' '.join(cleaned_words)
        
        return cleaned_translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation Error"

# Gradio Interface Function (Combining All Elements)
def process_image(image):
    """
    Processes the input image to generate a bilingual caption and color palette.
    
    Args:
        image (PIL.Image.Image or numpy.ndarray): The input image.
    
    Returns:
        tuple: Contains bilingual caption, hex color codes, palette image, and resized image.
    """
    # Ensure input is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB format
    image_rgb = image.convert("RGB")
    
    # Load and resize the image
    resized_image_np = load_image(image_rgb)
    resized_image_pil = Image.fromarray(resized_image_np)
    
    # Generate caption using the caption pipeline
    caption = generate_caption(image_rgb)
    
    # Translate caption to Arabic using the translation pipeline
    caption_arabic = translate_to_arabic(caption)
    
    # Extract dominant colors from the image
    colors = extract_colors(resized_image_np, k=8)
    color_palette = display_palette(colors)
    
    # Create palette image
    palette_image = create_palette_image(colors)
    
    # Combine English and Arabic captions
    bilingual_caption = f"English: {caption}\nArabic: {caption_arabic}"
    
    return (
        bilingual_caption,
        ", ".join(color_palette),
        palette_image,
        resized_image_pil
    )

# Create Gradio Interface using Blocks and add a submit button
with gr.Blocks(
    css=".gradio-container { height: 1000px !important; }"
) as demo:
    # Title and Description
    gr.Markdown(
        "<h1 style='text-align: center;'>"
        "Palette Generator from Image with Image Captioning"
        "</h1>"
    )
    gr.Markdown(
        """
        <p style='text-align: center;'>
        Upload an image or select one of the example images below to generate
        a color palette and a description of the image in both English and Arabic.
        </p>
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            # Image Input Component
            image_input = gr.Image(
                type="pil",
                label="Upload your image or select an example below"
            )
            # Submit Button
            submit_button = gr.Button("Submit")
            # Examples Component using Image URLs directly
            gr.Examples(
                examples=image_examples,  # List of lists with image URLs
                inputs=image_input,
                label="Example Images",
                examples_per_page=10,  # Adjust as needed
                fn=None,  # No need to specify a function since we're using URLs
            )
        with gr.Column(scale=1):
            # Output Components
            caption_output = gr.Textbox(
                label="Bilingual Caption",
                lines=5,
                max_lines=10
            )
            palette_hex_output = gr.Textbox(
                label="Color Palette Hex Codes",
                lines=2
            )
            palette_image_output = gr.Image(
                type="pil",
                label="Color Palette"
            )
            resized_image_output = gr.Image(
                type="pil",
                label="Resized Image"
            )
    
    # Define the action on submit button click
    submit_button.click(
        fn=process_image,
        inputs=image_input,
        outputs=[
            caption_output,
            palette_hex_output,
            palette_image_output,
            resized_image_output
        ],
    )

# Launch Gradio Interface
demo.launch()