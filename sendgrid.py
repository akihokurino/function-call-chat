import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


async def send_email(to: str, subject: str, body: str) -> None:
    payload = {
        "personalizations": [
            {
                "to": [{"email": to}],
                "subject": "Hello, World!2",
            }
        ],
        "from": {"email": "noreply-expertnetwork@liebra-inc.com"},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }

    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
    }

    print(payload)
    print(header)

    async with aiohttp.ClientSession() as session:
        async with session.post(
                SENDGRID_URL,
                headers=header,
                json=payload,
        ) as response:
            response_text: str = await response.text()
            if response.status != 202:
                raise ValueError(
                    f"SendGrid API Error {response.status}: {response_text}"
                )
