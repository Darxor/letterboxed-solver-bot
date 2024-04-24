import asyncio
import logging
import sys

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot import bot, main_router, default_router


async def main() -> None:
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(main_router)
    dp.include_router(default_router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    asyncio.run(main())
