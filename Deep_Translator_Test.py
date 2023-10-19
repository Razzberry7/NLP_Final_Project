from deep_translator import GoogleTranslator

# Translate from English to Korean
translated_english = GoogleTranslator(source='en', target='ko').translate("keep it up, you are awesome")  # output -> 계속해, 넌 정말 대단해

print(translated_english)

# Translate from Korean to English
translated_korean = GoogleTranslator(source='ko', target='en').translate("이해하지 않으면 후회할 거예요!")  # output -> If you don't understand, you'll regret it!

print(translated_korean)

