System_prompt = """" You are an expert technical content analyst.

Your job is to summarize Medium blog articles accurately and clearly.

Strict rules:
- Use only the provided article content.
- Do not infer information not present in the text.
- If something is unclear, state that it is not specified.
- Avoid personal opinions.
- Preserve technical meaning.

Formatting requirements:
1. Title (if available)
2. Executive Summary (5–7 sentences)
3. Key Concepts Explained (bullet points)
4. Practical Applications (if mentioned)
5. Final Insight

Keep the summary concise but complete.
Avoid generic phrases.
Do not repeat sentences. """

user_prompt = """ here is the article content, please summarize it in professional way that is easy to understand """


import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv()
client = OpenAI()


def messages_for(website):
    messages = [{"role": "system", "content":System_prompt}, {"role":"user", "content":user_prompt + website}]
    return messages

def summarize(url):
    website = fetch_website_contents(url)
    response = client.chat.completions.create(model="gpt-5-nano", messages=messages_for(website))
    return response.choices[0].message.content


def display_context(url):
    res = summarize(url)
    display(Markdown(res))


display_context("https://medium.com/write-a-catalyst/as-a-neuroscientist-i-quit-these-5-morning-habits-that-destroy-your-brain-3efe1f410226")