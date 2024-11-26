import openai
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are the trade-offs around deadwood in forests?"}]
)
print(response)