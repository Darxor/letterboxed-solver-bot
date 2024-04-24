import os

from aiogram import Bot, enums
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

bot = Bot(
    token=TELEGRAM_API_TOKEN,
    default=DefaultBotProperties(parse_mode=enums.ParseMode.HTML),
)
