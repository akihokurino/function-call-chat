import json
import os
from typing import final, AsyncGenerator, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
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

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

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


async def _chat_completion_stream(
    payload: _ChatCompletionPayload,
) -> AsyncGenerator[str, None]:
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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            for tool_call in chunk.choices[0].delta.tool_calls:
                if tool_call.function is None or tool_call.id is None:
                    continue

                print(tool_call.function.name)
                print(tool_call.function.arguments)

                if (
                    tool_call.function.name == "get_weather"
                    and tool_call.function.arguments
                ):
                    args = json.loads(tool_call.function.arguments)
                    location = args.get("location")
                    if not location:
                        yield "エラー: location が指定されていません。"
                        return

                    geocoded = await geocode_location(location)
                    if geocoded is None:
                        yield f"エラー: '{location}' の緯度経度が取得できませんでした。"
                        return
                    latitude, longitude = geocoded

                    weather_result = await fetch_weather(latitude, longitude)
                    if not weather_result.feature:
                        yield "エラー: 天気情報が取得できませんでした。"
                        return

                    rainfall = (
                        weather_result.feature[0]
                        .property_.weather_list.weather[0]
                        .rainfall
                    )
                    weather_info = f"{location} の降水量: {rainfall} mm"

                    messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=weather_info,
                        )
                    )

                    final_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=tools,
                        stream=True,
                    )

                    for final_chunk in final_response:
                        if final_chunk.choices and final_chunk.choices[0].delta.content:
                            yield final_chunk.choices[0].delta.content

        elif chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.post("/chat_completion")
async def _chat_completion(
    payload: _ChatCompletionPayload,
) -> StreamingResponse:
    return StreamingResponse(_chat_completion_stream(payload), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
