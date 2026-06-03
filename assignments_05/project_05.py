from dotenv import load_dotenv
from openai import OpenAI
import json


# Task 1: System Prompt Creation


load_dotenv()
client = OpenAI()



def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


system_prompt = """
You are a concise, professional job application coach for junior-to-mid-level candidates.
Help users improve job application materials, including resume bullets, cover letter openings, and answers to application-related questions.

Match the candidate's stated qualifications to the job requirements when possible.
Do not invent experience, credentials, metrics, companies, or results that the user did not provide.
Ask clarifying questions when important details are missing.
Stay focused on job application materials.

Always remind the user to review and edit any generated output before submitting it.
Acknowledge that you may not know the user's specific industry norms, and that the user should use their own judgment.

"""

# I made a delibarate choice to make the assistant stay accurate and evendence based. Also, to focus on more on junior to mid level positions.


# Task 2: Bullet Point Rewriter


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list, no other text. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result

    response_text = get_completion(messages)

    # avoiding common issues with LLM-generated JSON by stripping whitespace and code fences, then parsing
    response_text = response_text.strip()

    if response_text.startswith("```json"):
        response_text = response_text.removeprefix(
            "```json").removesuffix("```").strip()
    elif response_text.startswith("```"):
        response_text = response_text.removeprefix(
            "```").removesuffix("```").strip()

    # parse the cleaned response text as JSON and handle potential parsing errors
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error parsing JSON")
        data = []
    for item in data:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print()

    return data


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# The original bullets are vague and too general. They don’t show clear impact or results and use weak verbs. The model improved them by using stronger action verbs, adding more detail, and showing impact and ownership.


# Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident and specific. Avoid clichés and generic phrases (e.g., “I am eager to…”, “I am excited to…”) and make each sentence specific to the person’s background and company's needs.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    cover_letter_opening = get_completion(messages)
    print(cover_letter_opening)
    return cover_letter_opening


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

generate_cover_letter(job_title, background)

# I chose these examples because they show career changers connecting past experience to a new role in a specific, confident way. The few-shot pattern helps control the tone, structure, and level of detail so the output feels tailored instead of generic.


# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    # print(result.results[0].categories)

    if flagged:
        print(
            f"Your input may contain inappropriate content. Please rephrase and try again.")
        return False

    return True


user_input = input("Enter a resume bullet point to rewrite: ")

if is_safe(user_input):
    print("Input is safe. Proceeding with rewriting...")
    rewrite_bullets([user_input])
else:
    print("Input is not safe. Please modify your input and try again.")

# Unsafe input test
user_input2 = "i hate my job"

if is_safe(user_input2):
    print("Input is safe. Proceeding with rewriting...")
    rewrite_bullets([user_input2])
else:
    print("Input is not safe. Please modify your input and try again.")


# Task 5: The chatbot loop

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print(
                "\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            if not raw_bullets:
                print("No bullet points entered.")
            else:
                rewritten = rewrite_bullets(raw_bullets)
                print("\nRewritten Bullet Points:")
                for item in rewritten:
                    print(f"- {item['improved']}")
                print()
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input(
                "Job Application Helper: What is the job title? ").strip()
            background = input(
                "Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter = generate_cover_letter(job_title, background)
            print("\nCover Letter Opening:\n")
            print(cover_letter)
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print(reply)

            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethic Reflection

# 1. What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?

# It's so crucial to review what the bot generates, becuase if we don't do that, first of all , it may become clear that it was bot generated which shows unauthenticity and lack of effort. Also, the bot may generate content that is inaccurate, exaggerated, or not aligned with the job seeker's actual experience and skills, which could lead to credibility issues or even disqualification from the job application process. It would show candidate's laziness and lck of genuine interest in the position, which could be a red flag for employers.


# 2. Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?

# The bot's training data likely includes a disproportionate amount of content from certain industries (like tech), communication styles (like corporate or startup culture), and cultural backgrounds (like Western norms). This could lead the bot to favor advice that aligns with those dominant patterns, potentially disadvantaging job seekers from underrepresented industries, non-Western cultures, or those who communicate in less conventional ways. For example, it might suggest resume formats or cover letter tones that are more common in tech or corporate settings, which may not resonate with employers in creative fields or in different cultural contexts.
