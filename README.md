# Gradio_tutorial

## Download
```python
!pip install gradio==3.50.2
```
Why we choose 3.50.2 rather than the latest version is because the latest version removes the tool parameter like "sources". <br>
So, there are some function we can't use easily because of this. <br>

## What is Gradio?
Gradio is an open-source Python library that simplifies the process of creating user interfaces for machine learning models and other Python functions. <br>
It allows developers to build interactive web-based interfaces that users can access through a browser. <br>
After you run your code, it will give you a url link, so you can use your app on the website! <br>

## Easy application of Gradio
### 1. Hello World!
```python
import gradio as gr

def greet(name):
    return "Hello " + name

textbox = gr.Textbox(label="Type your name here:", placeholder="Jessica", lines=2)

gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
```
Gradio's interface has three inportant parameters: 
Interface(fn, inputs, outputs, ...) 
* fn: the prediction function that is wrapped by the Gradio interface. 
* inputs: the input component type. 
* outputs: the output component type. 
 <br>
And this is the result:


https://github.com/user-attachments/assets/e654d6f7-a7c7-4cf9-8a10-7b786d4cc97f


### 2. Make prediction!
Using the pipeline() function from HuggingFace Transformers. 
```python
from transformers import pipeline

model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion
gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```
This function completes prompts that you provide. <br>
 <br>
And this is the result:


https://github.com/user-attachments/assets/241196bd-89ee-49c4-9318-636292bb2aaa



### 3. Reverse Audio
Build an Interface that works with audio.
```python
import numpy as np
import gradio as gr

def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(source="microphone" ,type="numpy", label="Speak here...")
gr.Interface(reverse_audio, mic, "audio").launch()
```
 <br>
And this is the result: 


https://github.com/user-attachments/assets/4e38f527-5c8b-458c-abde-6d42c37d6866


### 4. Easy speech to text
Load speech recognition model using the pipeline() function from HuggingFace Transformers. 
```python
from transformers import pipeline
import gradio as gr

model = pipeline("automatic-speech-recognition")


def transcribe_audio(mic=None, file=None):
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must either provide a mic recording or a file"
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(source="microphone", type="filepath", optional=True),
        gr.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
).launch()
```
 <br>
And this is the result:


https://github.com/user-attachments/assets/b5665c3b-174e-4f89-9db3-96a1ead6ab52


### 5. Pictionary
The Interface class supports some optional parameters: 
* title: you can give a title to your demo, which appears above the input and output components. 
* description: you can give a description for the interface, which appears above the input and output components and below the title. 
* article: you can also write an expanded article explaining the interface. 
* theme: Set the theme to use one of default, huggingface, grass, peach. 
* examples: provide some example inputs for the function. These appear below the UI components and can be used to populate the interface. 
* live: to make your model reruns every time the input changes, you can set live=True. <br>
We use class_names.txt and pytorch_model.bin that HuggingFace provided: <br>
class_names.txt <br>
``
airplane
alarm_clock
anvil
apple
axe
baseball
baseball_bat
basketball
beard
bed
bench
bicycle
bird
book
bread
bridge
broom
butterfly
camera
candle
car
cat
ceiling_fan
cell_phone
chair
circle
clock
cloud
coffee_cup
cookie
cup
diving_board
donut
door
drums
dumbbell
envelope
eye
eyeglasses
face
fan
flower
frying_pan
grapes
hammer
hat
headphones
helmet
hot_dog
ice_cream
key
knife
ladder
laptop
light_bulb
lightning
line
lollipop
microphone
moon
mountain
moustache
mushroom
pants
paper_clip
pencil
pillow
pizza
power_outlet
radio
rainbow
rifle
saw
scissors
screwdriver
shorts
shovel
smiley_face
snake
sock
spider
spoon
square
star
stop_sign
suitcase
sun
sword
syringe
t-shirt
table
tennis_racquet
tent
tooth
traffic_light
tree
triangle
umbrella
wheel
wristwatch
`` <br>
And here is the code: 
```python
from pathlib import Path
import torch
import gradio as gr
from torch import nn

LABELS = Path("class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(x)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)
    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

interface = gr.Interface(
    predict,
    inputs=gr.Sketchpad(brush_width=1),
    outputs="label",
    theme="huggingface",
    title="Sketch Recognition",
    description="Who wants to play Pictionary? Draw a common object like a shovel or a laptop, and the algorithm will guess in real time!",
    article="<p style='text-align: center'>Sketch Recognition | Demo Model</p>",
    live=True,
)
interface.launch()
```
<br>
And this is the result:


https://github.com/user-attachments/assets/f9a42907-8e42-4801-9aee-36117770630b


We can see that this model is not that precise. Maybe we can try an other.
### 6. Chat Bot with OpenAI API
Of course, we can use OpenAI like Chatgpt.
```python
!pip install openai==0.27.0

import openai
import gradio as gr

openai.api_key = "OpenAI_API_key"

def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        message = response['choices'][0]['message']['content']
        return message.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# gpt-3.5-turbo Chatbot")
    with gr.Row():
        txt_input = gr.Textbox(label="Your Message", placeholder="Ask me anything!")
        txt_output = gr.Textbox(label="Chatbot Response")
    submit_btn = gr.Button("Send")

    submit_btn.click(chat_with_gpt, inputs=txt_input, outputs=txt_output)

demo.launch()
```
<br>
And this is the result:


https://github.com/user-attachments/assets/b7019dc3-87ff-4a4d-ab4a-f2ed46bb8701


## What else can we do?
We can use Gradio as our interactive demo base and we use others llm api. <br>
For example:
* OpenAI
* Assembly.AI-Speech to text
* elevenlabs.io-Text to speech
* HuggingFace-from transformers import pipeline <br>
<br>
And we can use langchain and coding by python.<br>
Compare Langchain and Langflow(what we use to make app in july)<br>
| **Feature**        | **LangChain**                           | **LangFlow**                               |
|--------------------|-----------------------------------------|---------------------------------------------|
| **Min threhold**   | May should good at coding               | Know how to code and have basic knowledge   |
| **Operation**      | coding                                  | many components, link them                  |
| **Flexibilty**     | Well                                    | not that much                               |


