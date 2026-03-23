import asyncio
import mlflow
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("api_key"))

SYSTEM_PROMPT = "Explain Quantum Physics to an 8-year-old using a simple analogy."


async def run_prompt_test(temp):
    with mlflow.start_run(run_name=f"Temp_{temp}", nested=True):
        try:
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": SYSTEM_PROMPT}],
                temperature=temp,
            )
            answer = response.choices[0].message.content
            mlflow.log_text(answer, "response.txt")
            mlflow.log_metric("length", len(answer))
            print(f"Done: Temp {temp}")
        except Exception as e:
            print(f"Error: {e}")


async def main():
    mlflow.set_experiment("Free_LLM_Testing")
    with mlflow.start_run(run_name="Groq_Llama3_Test"):
        await asyncio.gather(*[run_prompt_test(t) for t in [0.2, 0.7, 1.2]])


if __name__ == "__main__":
    asyncio.run(main())
