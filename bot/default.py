from aiogram import Router, types, F

default_router = Router()

@default_router.message(F.document)
async def document_handler(message: types.Message) -> None:
    await message.reply(
        "Пожалуйста, пришли скриншот как фото (со сжатием), а не как документ"
    )

@default_router.message()
async def default_handler(message: types.Message) -> None:
    await message.reply(
        "Я понимаю только сообщения с картинками :(\n\nПришли скриншот из letterboxed как фото (не как документ)"
    )