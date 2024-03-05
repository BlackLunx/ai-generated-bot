from config_reader import config

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram import F

from model.ai_check import check_text

import asyncio

dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет, данный бот умеет проверять текст на сгенерированность, для этого отправь сообщение с текстом в чат боту!")

mapping = {
    True: 'Данный текст является сгенерированным!',
    False: 'Данный текст не является сгенерированным!',
}

@dp.message(F.text)
async def cmd_add_to_list(message: types.Message):
    answer = await check_text(message.text)
    await message.answer(mapping[answer])

async def bot_start():
    bot = Bot(config.bot_token.get_secret_value())
   
    # Запускаем бота и пропускаем все накопленные входящие
    # Да, этот метод можно вызвать даже если у вас поллинг
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)



if __name__ == "__main__":
    asyncio.run(bot_start())