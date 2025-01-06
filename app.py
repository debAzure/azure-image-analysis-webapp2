import os
import sys
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import requests
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_image(image_data):
    print('\nAnalyzing image...')
    try:
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.TAGS, VisualFeatures.OBJECTS, VisualFeatures.PEOPLE],
        )

        analysis_results = {}

        # Get captions
        if result.caption:
            analysis_results['caption'] = result.caption.text

        # Get dense captions
        analysis_results['dense_captions'] = [caption.text for caption in result.dense_captions.list]

        # Get tags
        analysis_results['tags'] = [tag.name for tag in result.tags.list]

        # Get objects and people
        objects = []
        people = []

        # Use BytesIO to read the image from binary data
        image = Image.open(BytesIO(image_data))
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        if result.objects:
            for detected_object in result.objects.list:
                objects.append(detected_object.tags[0].name)
                r = detected_object.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

        if result.people:
            for detected_person in result.people.list:
                people.append(f'Person (confidence: {detected_person.confidence * 100:.2f}%)')
                r = detected_person.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

        # Save annotated image
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_image.jpg')
        plt.imshow(image)
        plt.tight_layout(pad=0)
        fig.savefig(annotated_image_path)
        analysis_results['objects'] = objects
        analysis_results['people'] = people

        return analysis_results, annotated_image_path
    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")
        return None, None

def remove_background(image_url):
    # Define the API version and mode
    api_version = "2023-02-01-preview"
    mode = "backgroundRemoval"

    print('\nRemoving background from image...')

    url = f"{ai_endpoint}computervision/imageanalysis:segment?api-version={api_version}&mode={mode}"
    headers = {
        "Ocp-Apim-Subscription-Key": ai_key,
        "Content-Type": "application/json"
    }
    body = {"url": image_url}
    response = requests.post(url, headers=headers, json=body)
    
    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: Received response code {response.status_code} for background removal")
        print(f"Response text: {response.text}")
        return None

    image = response.content

    # Check if the image content is valid
    try:
        img = Image.open(BytesIO(image))
        img.verify()  # Verify the image file
        print("Image is valid")
    except Exception as e:
        print(f"Error: Invalid image data for background removal: {e}")
        return None

    # Save image
    background_file = os.path.join(app.config['UPLOAD_FOLDER'], 'background_removed.png')
    print(f"Saving background removed image at: {background_file}")
    with open(background_file, "wb") as file:
        file.write(image)
    
    return background_file

def foreground_matting(image_url):
    # Define the API version and mode
    api_version = "2023-02-01-preview"
    mode = "foregroundMatting"

    print('\nPerforming foreground matting on image...')

    url = f"{ai_endpoint}computervision/imageanalysis:segment?api-version={api_version}&mode={mode}"
    headers = {
        "Ocp-Apim-Subscription-Key": ai_key,
        "Content-Type": "application/json"
    }
    body = {"url": image_url}
    response = requests.post(url, headers=headers, json=body)

    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: Received response code {response.status_code} for foreground matting")
        print(f"Response text: {response.text}")
        return None

    image = response.content

    # Check if the image content is valid
    try:
        img = Image.open(BytesIO(image))
        img.verify()  # Verify the image file
        print("Image is valid")
    except Exception as e:
        print(f"Error: Invalid image data for foreground matting: {e}")
        return None

    # Save image
    foreground_file = os.path.join(app.config['UPLOAD_FOLDER'], 'foreground_matted.png')
    print(f"Saving foreground matted image at: {foreground_file}")
    with open(foreground_file, "wb") as file:
        file.write(image)
    
    return foreground_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_data = open(filepath, "rb").read()

        # Analyze the image
        analysis_results, annotated_image = analyze_image(image_data)

        # Perform background removal and foreground matting
        background_file = remove_background(filepath)
        foreground_file = foreground_matting(filepath)

        # Debugging: print paths to confirm they are correct
        print(f"Annotated Image Path: {annotated_image}")
        print(f"Background Removed Image Path: {background_file}")
        print(f"Foreground Matting Image Path: {foreground_file}")

        return render_template('index.html', 
                               analysis_results=analysis_results, 
                               annotated_image=annotated_image, 
                               background_file=background_file, 
                               foreground_file=foreground_file, 
                               image_url=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
