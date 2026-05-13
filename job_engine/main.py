import asyncio
import os
import random
import json
import logging
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from browser_use import Agent, Browser, Controller

from .memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory manager
memory = MemoryManager(data_dir=os.path.join(os.path.dirname(__file__), "data"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0
)

# Initialize controller for custom actions
controller = Controller()

# Global state to track if we need to abort the current job application
abort_current_job = False

@controller.action("Check knowledge base for an answer to a form field or screening question. You MUST call this before filling out ANY field.")
def check_knowledge_base(question_text: str, current_url: str) -> str:
    """
    Checks the answers.json knowledge base to see if we know how to answer this question.
    """
    global abort_current_job

    answers_db = memory.load_answers()

    # Use a quick LLM call to evaluate if the answer exists in the DB with high confidence
    prompt = f"""
You are a strict data matching assistant.
Here is the applicant's knowledge base:
{json.dumps(answers_db, indent=2)}

Question from job application: "{question_text}"

Does the knowledge base contain the answer to this question?
If you are >= 95% confident the answer is present, return ONLY the exact value/answer from the knowledge base.
If the answer is missing, incomplete, or you are <95% confident, return EXACTLY the string "NULL".
Do not explain your reasoning.
"""
    try:
        response = llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"Error querying LLM in knowledge base check: {e}")
        response = "NULL"

    if response == "NULL":
        logger.info(f"Unknown field detected: '{question_text}'. Logging and aborting.")
        memory.log_pending_question(question_text, current_url)
        abort_current_job = True
        return "UNKNOWN_FIELD - You must immediately exit and stop processing this task."

    return f"KNOWN_ANSWER: {response}"

@controller.action("Wait for a random delay between 2 and 5 seconds to simulate human review time")
async def human_jitter_delay():
    """
    Call this before transitioning pages or submitting forms to mimic human behavior.
    """
    delay = random.uniform(2, 5)
    logger.info(f"Applying jitter delay of {delay:.2f} seconds...")
    await asyncio.sleep(delay)
    return f"Waited {delay:.2f} seconds."

async def process_job(url: str, browser: Browser):
    global abort_current_job
    abort_current_job = False

    logger.info(f"Processing job application at: {url}")

    system_prompt = f"""
You are a human-like job application assistant. Your goal is to fill out the job application at the current URL.
You must adhere to these STRICT rules:
1. Before filling out ANY input field, dropdown, or checkbox, you MUST use the `check_knowledge_base` tool. Pass the exact label/question text and the current URL.
2. If `check_knowledge_base` returns a string starting with 'UNKNOWN_FIELD', you MUST immediately stop all actions, do not submit the form, and mark your task as complete with a failure message.
3. If you have all required answers and are ready to click 'Submit' or move to the 'Next Page', you MUST call the `human_jitter_delay` tool first.
4. Do not guess any information. If a field isn't in the knowledge base, it is an UNKNOWN_FIELD.
"""

    agent = Agent(
        task=f"Navigate to {url}, fill out the application using the knowledge base, and submit it.",
        llm=llm,
        browser=browser,
        controller=controller,
        system_prompt=system_prompt,
    )

    try:
        history = await agent.run()
        if abort_current_job:
            logger.warning(f"Aborted application for {url} due to unknown field.")
        else:
            logger.info(f"Successfully processed {url}")
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")

async def main():
    job_queue = memory.load_job_queue()
    if not job_queue:
        logger.info("Job queue is empty.")
        return

    # Use raw string for Chrome profile path as requested
    profile_path = r"C:\Users\laksh\AppData\Local\Google\Chrome\User Data\Profile 8"

    # Initialize Browser for non-headless mode and specific profile
    # For browser-use, we can pass args directly to Playwright using the 'args' parameter.
    # We will pass the user data dir through args.
    browser = Browser(
        headless=False,
        args=[f"--user-data-dir={profile_path}"]
    )

    for url in job_queue:
        await process_job(url, browser)

    await browser.close()
    logger.info("Finished processing job queue.")

if __name__ == "__main__":
    asyncio.run(main())
