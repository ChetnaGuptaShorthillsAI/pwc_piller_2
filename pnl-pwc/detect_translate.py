from mtranslate import translate

class Translator:
    def detect_english_language(self, input_string):
        # Detect if the language is English by translating it to English
        if input_string is None:
            # Handle the case where input_string is None
            return True
        if isinstance(input_string, (int, float)):
            # Handle the case where input_string is an integer or float
            return True
        
        translated_text = translate(input_string, 'en')
        # Compare the translated text with the input text
        if translated_text.lower() == input_string.lower():
            return True
        else:
            return False

    def convert_to_english(self, input_string):
        # Translate the input string to English
        if input_string is None:
            # Handle the case where input_string is None
            return None
        if isinstance(input_string, int):
        # Handle the case where input_string is an integer
            return str(input_string)
        if isinstance(input_string, float):
            # Handle the case where input_string is a float
            return str(input_string)
        translated_text = translate(input_string, 'en')
        return translated_text

# Example usage
# if __name__ == "__main__":
#     translator = Translator()

#     input_text = "121231234"
#     is_english = translator.detect_english_language(input_text)
#     print(is_english)
#     if(is_english==False):
#         translated_text = translator.convert_to_english(input_text)
#         print(translated_text)


