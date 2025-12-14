from google import genai
import os

client = genai.Client(api_key=os.getenv("HTT_RATIO_GEMINI_API_KEY"))

uploaded_file = client.files.upload(file = "reports\\shell-annual-report-2024.pdf")

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = ["I want to use the information provided here to calculate the production growth rate which is a ratio between the current period production minus the prior period production all divided by the prior period production. This is of course in the context of oil and gas production. Can you find this data and calculate the ratio for me?", uploaded_file]
)

print(response.text)