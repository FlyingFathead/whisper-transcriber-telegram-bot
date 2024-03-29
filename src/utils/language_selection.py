# language_selection.py
# (work in progress!)

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Define a comprehensive list of supported languages and their emojis.
LANGUAGES = {
    'en': '🇬🇧 English',
    'fi': '🇫🇮 Finnish',
    'fr': '🇫🇷 French',
    'es': '🇪🇸 Spanish',
    'zh': '🇨🇳 Chinese',
    # more supported languages to come...
}

ITEMS_PER_PAGE = 5

def build_menu(language_buttons, n_cols, header_buttons=None, footer_buttons=None):
    menu = [language_buttons[i:i + n_cols] for i in range(0, len(language_buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu

async def ask_language(bot, chat_id, page=0):
    language_buttons = []
    start_index = page * ITEMS_PER_PAGE
    language_subset = list(LANGUAGES.items())[start_index:start_index + ITEMS_PER_PAGE]

    for code, language in language_subset:
        button_text = f"{language} ({code})"
        language_buttons.append(InlineKeyboardButton(button_text, callback_data=code))

    navigation_buttons = []
    if page > 0:
        navigation_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f'prev_{page - 1}'))
    if start_index + ITEMS_PER_PAGE < len(LANGUAGES):
        navigation_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f'next_{page + 1}'))

    keyboard = build_menu(language_buttons, 2, footer_buttons=navigation_buttons)
    reply_markup = InlineKeyboardMarkup(keyboard)
    await bot.send_message(chat_id, "Please select the language for transcription:", reply_markup=reply_markup)

