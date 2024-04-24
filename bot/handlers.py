import io
import logging
import random

from aiogram import Bot, F, Router, types

from ocr import process_image
from solver import solve

main_router = Router()


@main_router.message(F.photo)
async def handle_docs_photo(message: types.Message, bot: Bot) -> None:
    file = await bot.get_file(message.photo[-1].file_id)
    image: io.BytesIO = await bot.download_file(file.file_path)

    try:
        ocr_image, ocr_text = await process_image(image)
    except Exception as e:
        logging.exception(e)
        await message.reply(f"Во время распознавания произошла ошибка: {e}")
        return

    reply_text = (
        f"Буквы твоего letterboxed: <pre>{ocr_text}</pre>\n\n приступаю к решению..."
        if ocr_text
        else "Буквы не найдены :(\n\nПопробуй снять скриншот иначе"
    )

    await message.reply_photo(
        types.BufferedInputFile(ocr_image.read(), "image_ocr.png"),
        caption=reply_text,
        parse_mode="HTML",
    )
    
    if not ocr_text:
        return

    try:
        n_solutions, solutions = solve(ocr_text)
    except Exception as e:
        logging.exception(e)
        await message.reply(f"Во время решения произошла ошибка: {e}")
        return

    if not n_solutions:
        await message.reply("Решение не найдено :(")
        return

    solution_text = "Решения за три слова:"

    random.seed(42)
    random.shuffle(solutions)
    solutions = solutions[:20]

    for i, solution in enumerate(solutions, start=1):
        solution = "-".join(w[0] for w in solution)
        solution_text += f"\n{i}. {solution}"

    await message.reply(solution_text)
