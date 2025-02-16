from typing import Optional

import aiohttp

GEO_API_URL = "https://nominatim.openstreetmap.org/search"


async def geocode_location(location: str) -> Optional[tuple[float, float]]:
    async with aiohttp.ClientSession() as session:
        params = {"q": location, "format": "json", "limit": "1"}
        async with session.get(GEO_API_URL, params=params) as response:
            data = await response.json()
            if data and isinstance(data, list) and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return lat, lon
    return None
