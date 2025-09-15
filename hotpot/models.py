import os
import openai
from openai import OpenAI


import backoff 
from transformers import GPT2Tokenizer
import warnings
from dotenv import load_dotenv, find_dotenv
from loguru import logger
import sys
client = None

# Configure loguru with colored output
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    f'{__name__}.log',
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


load_dotenv(find_dotenv())


completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

llm2use = os.getenv("LLM", "custom")
if llm2use == "openai":
    logger.info("Using OpenAI")
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    api_base = os.getenv("OPENAI_API_BASE", "")
    
    
elif llm2use == "custom":
    logger.info(f"Using Custom LLM")
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "http://69.48.159.10:30003/v1")
    model = os.getenv("OPENAI_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")

    logger.debug(f"LLM API Base: {api_base}, Model: {model}")

else:
    logger.error(f"Invalid LLM: {llm2use}, can only be openai or custom")
    raise ValueError("Invalid LLM: {}, can only be openai or custom".format(llm2use))

client = OpenAI(api_key=api_key, base_url=api_base)

def tokens_in_text(text):
    """
    Accurately count the number of tokens in a string using the GPT-2 tokenizer.
    
    :param text: The input text.
    :return: The exact number of tokens in the text.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
    return len(tokens)


@backoff.on_exception(backoff.expo, openai.APIConnectionError)
def completions_with_backoff(**kwargs):
    global client
    
    return client.chat.completions.create(**kwargs)
# 🔴 this function is called from outside
# custom logic will be ingected here
def gpt(prompt, model="llama-3.1-70b", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="llama-3.1-70b", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens, client
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens, client
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    else:
        cost = "infinite 💵"
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
