import json
import os
from typing import Literal, final, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from geo import geocode_location
from yahoo import fetch_weather

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された場所（地名）の現在の天気を取得します。必ず location に地名を入れてください。（例: location='駒沢'）",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "場所の名前（例: 駒沢, 東京, ロンドン）。ユーザーの質問に含まれる都市名をそのまま使ってください。",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
model = "gpt-4o"

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@final
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


@final
class _ChatCompletionPayload(BaseModel):
    messages: list[Message]


def _function_call_message(
    response: ChatCompletion,
    messages: list[ChatCompletionMessageParam],
    tool_id: str,
    info_from_tool: str,
) -> ChatCompletion:
    _assistant_message = response.choices[0].message
    assistant_message = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=_assistant_message.content,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=cast(Function, tool_call.function),
                type="function",
            )
            for tool_call in (_assistant_message.tool_calls or [])
        ],
    )
    messages.append(assistant_message)  # **
    messages.append(
        ChatCompletionToolMessageParam(
            role="tool",
            content=info_from_tool,
            tool_call_id=tool_id,
        )
    )

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    return final_response


@app.post("/chat_completion")
async def _chat_completion(payload: _ChatCompletionPayload) -> JSONResponse:
    messages: list[ChatCompletionMessageParam] = []
    for message in payload.messages:
        if message.role == "assistant":
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
                )
            )
        if message.role == "user":
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=message.content,
                )
            )

    # Stream=trueだとfunction callがうまく動かない
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    if response.choices and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function is None or tool_call.id is None:
            return JSONResponse({"error": "Function call error"}, status_code=400)

        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")

        if tool_call.function.name == "get_weather":
            args = json.loads(tool_call.function.arguments)
            location = args.get("location")
            if not location:
                return JSONResponse(
                    {"error": "location が指定されていません。"}, status_code=400
                )
            print(f"Location: {location}")

            geocoded = await geocode_location(location)
            if geocoded is None:
                return JSONResponse(
                    {"error": f"'{location}' の緯度経度が取得できませんでした。"},
                    status_code=400,
                )
            latitude, longitude = geocoded
            print(f"Latitude: {latitude}, Longitude: {longitude}")

            weather_result = await fetch_weather(latitude, longitude)
            rainfall = (
                weather_result.feature[0].property_.weather_list.weather[0].rainfall
            )
            weather_info = f"{location} の降水量: {rainfall} mm"
            print(f"Weather: {weather_info}")

            final_response = _function_call_message(
                response, messages, tool_call.id, weather_info
            )

            return JSONResponse({"response": final_response.choices[0].message.content})

    return JSONResponse({"response": response.choices[0].message.content})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
