from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model = "gpt-4o-mini-2024-07-18",
  messages = [
    {"role": "user", "content": "こんにちは。何か面白いことを言ってください。"}
  ],
  temperature=0.7,
  max_tokens=2000,
  top_p=0.95
)

print(response.choices[0].message.content)
