from google import genai
import os
from typing import Any
import json
from google.genai import types

def production_growth_rate(file_path: str) -> dict[str, Any]:
    client = genai.Client(api_key=os.getenv("HTT_RATIO_GEMINI_API_KEY"))

    uploaded_file = client.files.upload(file = file_path)

    json_schema = {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "year": {"type": "integer"},
            "quarter": {"type": "integer"},
            "production_growth_rate": {"type": "number"}
        },
        "required": ["company_name", "year", "quarter", "production_growth_rate"]
    }

    response = client.models.generate_content(
        model = "gemini-2.5-flash",

        contents = ["I want to use the information provided here to calculate the (upstream) production growth rate (QoQ only) \
            which is a ratio between the current period production minus the prior period production all divided by the prior period \
            production. This is of course in the context of oil and gas production. Can you find this data and \
            calculate the ratio for me? Provide the information in a JSON string only. \
            An example is given: {'company_name': 'Shell', 'year': 2024, 'quarter': 1, 'production_growth_rate': 1.2}", uploaded_file
            ],

        #contents = ["I want to use the information provided here to calculate the production growth rate which is a ratio \
        #between the current period production minus the prior period production all divided by the prior period production. \
        #This is of course in the context of oil and gas production. \
        #Can you find this data and calculate the ratio for me? \
        #Provide the information in a JSON string only. An example is given: \
        #    {'company_name': 'Shell', 'year': 2024, 'quarter': 1, 'production_growth_rate': 1.2}", uploaded_file
        #],

        config = types.GenerateContentConfig(
            response_mime_type = "application/json",
            response_schema = json_schema
        )
    )

    response_json = response.text
    response_json = json.loads(response_json)
    
    return response_json

def save_production_growth_rate_data(data: dict[str, Any], filename: str) -> None:
    pass    


if __name__ == "__main__":
    test_dict = production_growth_rate("reports\quarterly\q3-2025-quarterly-press-release.pdf")
    print(test_dict)