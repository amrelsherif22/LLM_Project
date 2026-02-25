system_prompt = f"""
you are a professional resume reviewer and suggester you will be given a bunch of jobs, that has a lot of details and has a description,
for these description, I want you to look over the experience I have in my resume that I will be Sending to you and check these jobs,
according to experience, target position should be New York,  matching skills, and if I would stand out in applying, you should be able to
adjust my resume based on a job, you either remove experience, rewrite it, or limit the cv to these experiences needed for each specific job
for each job that I'm a high match should come with saying for this specific job make your job experience bullet points like the below in your response
you should immedietly return a full tailored experience adjustments either new, remove, or keep on each job recommended"""
with open("experience.txt") as r:
    text = r.read()
    # Replace problematic bullet and dash characters that appear as 'â€¢' and 'â€“'
    text = text.replace('â€¢', '•').replace('â€”', '-').replace("â€“", "-")
    resume = text.encode("utf-8", errors="replace").decode("utf-8").splitlines()
resume_str = "\n".join(resume)
resume_str
from linkedin_scraper import print_jobs,fetch_linkedin_jobs
LINKEDIN_URL = (
        "https://www.linkedin.com/jobs/search/?alertAction=viewjobs&currentJobId=4374547986&f_TPR=r8600&geoId=105080838&keywords=Software%20Engineer&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R"
    )

jobs =fetch_linkedin_jobs(
        LINKEDIN_URL,
        max_jobs=50,  # Start with fewer for testing
        include_description=True,
        delay=1.0,
        desc_delay=0.8,
    )
list_of_jobs = []
for job in jobs:
    list_of_jobs.append(job.print_full())

user_prompt = f"""
I want you to filter out these jobs match with years of experience, skills, and location of new york and return the adjustment needed immediatly for my resume as bullet points to be used on my resumse as new, remove, keepy, 
only for high matches that i will get a chance to be intereviewed
 without asking me that I got from a recent search on LinkedIn. We are looking for description or requirement,
or both, from all the contents of the job that I am sending you so you can understand what they are looking for. Here is a list of all the jobs I have found: {list_of_jobs}
You should be able to match me and adjust my resume per job if I'm a good match so I stand out.
Here is my resume:
{resume_str}
"""
user_prompt
messages = [{"role":"system", "content":"system_prompt"}, {"role":"user", "content": user_prompt}]
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI
openai = OpenAI()

def get_job():
    stream = openai.chat.completions.create(model="gpt-5-mini", messages=messages, stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
import gradio as gr

gr.Interface(
    fn=get_job,
    inputs=None,  # no user inputs, just a button
    outputs=gr.Markdown(label="Response"),
    flagging_mode="never",
).launch()
