import os
from typing import Optional, final, Any

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
YAHOO_API_CLIENT_ID = os.getenv("YAHOO_API_CLIENT_ID")
YAHOO_API_URL = "https://map.yahooapis.jp/weather/V1/place"


@final
class ResultInfo(BaseModel):
    count: int = Field(alias="Count")
    total: int = Field(alias="Total")
    start: int = Field(alias="Start")
    status: int = Field(alias="Status")
    latency: float = Field(alias="Latency")
    description: Optional[str] = Field(alias="Description", default=None)

    model_config = {"populate_by_name": True}


@final
class Geometry(BaseModel):
    type_: str = Field(alias="Type")
    coordinates: str = Field(alias="Coordinates")

    model_config = {"populate_by_name": True}


@final
class Weather(BaseModel):
    type_: str = Field(alias="Type")
    date: str = Field(alias="Date")
    rainfall: float = Field(alias="Rainfall")

    model_config = {"populate_by_name": True}


@final
class WeatherList(BaseModel):
    weather: list[Weather] = Field(alias="Weather")

    model_config = {"populate_by_name": True}


@final
class Property(BaseModel):
    weather_area_code: int = Field(alias="WeatherAreaCode")
    weather_list: WeatherList = Field(alias="WeatherList")

    model_config = {"populate_by_name": True}


@final
class Feature(BaseModel):
    id_: str = Field(alias="Id")
    name: str = Field(alias="Name")
    geometry: Geometry = Field(alias="Geometry")
    property_: Property = Field(alias="Property")

    model_config = {"populate_by_name": True}


@final
class WeatherData(BaseModel):
    result_info: ResultInfo = Field(alias="ResultInfo")
    feature: list[Feature] = Field(alias="Feature")

    model_config = {"populate_by_name": True}


async def fetch_weather(lat: float, lng: float) -> WeatherData:
    payload = {
        "coordinates": f"{lng},{lat}",
        "output": "json",
        "appid": str(YAHOO_API_CLIENT_ID),
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(YAHOO_API_URL, params=payload) as response:
            response.raise_for_status()
            data: Any = await response.json()

    try:
        weather_data = WeatherData.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"レスポンスのマッピングに失敗しました: {e}")

    return weather_data
