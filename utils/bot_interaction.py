"""Здесь должен быть удобный интерфейс взаимодействия с ботом"""
import telebot
from config import API_TOKEN



bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def handle_message(message):
    print(message.from_user.id)

bot.infinity_polling()