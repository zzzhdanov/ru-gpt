import eel
import random
from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline

model_name = 'ai-forever/rugpt3small_based_on_gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = {
    "max_length": random.randint(250, 400),
    "temperature": 1.1,
    "top_p": 2.,
    "num_beams": 10,
    "repetition_penalty": 1.5,
    "num_return_sequences": 9,
    "no_repeat_ngram_size": 2,
    "do_sample": True
}


generation = pipeline('text-generation', model=model,
                      tokenizer=tokenizer, device=-1)


@eel.expose
def generate_response(input):
    return generation(input, **config)[0]['generated_text']

eel.init("web")
eel.start("main.html")
