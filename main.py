import streamlit as st
from transformers import pipeline
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
import ollama

def image_Caption(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    inputs = processor(images=image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def detect_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_size = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_size, threshold=0.9)[0]

    detects = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detects += f'Bounding Box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}] | Class: {model.config.id2label[int(label)]} | Score: {float(score):.2f}\n'
    return detects

def load_classifier():
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def chatbot():
    st.title("Conversational Image Recognition Chatbot")
    st.write("Upload an image and ask any question about it!")

    classifier = load_classifier()

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image_data = uploaded_image.read() 
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
            return

        user_input = st.text_input("Ask a question about the image:")

        if user_input:
            try:
                response = ollama.chat(model='gemma:2b', messages=[
                    {
                        'role': 'user',
                        'content': user_input,
                    },
                ])

                if 'message' in response:
                    response_content = response['message']['content']
                    st.write(response_content)

                    task = classifier(
                        user_input,
                        candidate_labels=["describe the image", "detect objects"],
                        multi_label=True
                    )

                    if "describe the image" in task['labels']:
                        caption = image_Caption(image_data)
                        st.write("Caption: ", caption)
                    elif "detect objects" in task['labels']:
                        detections = detect_image(image_data)
                        st.write("Detections:\n", detections)
                    else:
                        st.write("Could not determine task from input.")
                else:
                    st.error("No valid response.")
            except Exception as e:
                st.error(f"Error with Ollama chat: {e}")
    else:
        st.write("Please upload an image to proceed.")
if __name__ == "__main__":
    chatbot()