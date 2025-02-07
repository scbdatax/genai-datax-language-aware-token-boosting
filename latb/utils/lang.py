import re

language_dict = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "km": "Khmer",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh_cn": "Chinese (Simplified)",
    "zh_hk": "Chinese (Hong Kong)",
    "zh_tw": "Chinese (Traditional)"
}

xlsum_lang_map = {
    'am': 'amharic',
    'ar': 'arabic',
    'az': 'azerbaijani',
    'bn': 'bengali',
    'my': 'burmese',
    'zh': 'chinese_simplified',
    'zt': 'chinese_traditional',
    'en': 'english',
    'fr': 'french',
    'gu': 'gujarati',
    'ha': 'hausa',
    'hi': 'hindi',
    'ig': 'igbo',
    'id': 'indonesian',
    'ja': 'japanese',
    'rn': 'kirundi',
    'ko': 'korean',
    'ky': 'kyrgyz',
    'mr': 'marathi',
    'ne': 'nepali',
    'om': 'oromo',
    'ps': 'pashto',
    'fa': 'persian',
    'pi': 'pidgin',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ru': 'russian',
    'gd': 'scottish gaelic',
    'sr': 'serbian (cyrillic)',
    'sl': 'serbian (latin)',
    'si': 'sinhala',
    'so': 'somali',
    'es': 'spanish',
    'sw': 'swahili',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'ti': 'tigrinya',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'yo': 'yoruba'
}

xlsum_prompt_template = {
  "am": "እባክዎ ጽሑፉን በአማርኛ ይዘምኑ። ጽሑፍ፡ {} ማጠቃለያ፡ ",
  "ar": "يرجى تلخيص النص باللغة العربية. النص: {} الملخص: ",
  "az": "Mətnin azərbaycanca xülasəsini verin. Mətn: {} Xülasə: ",
  "bn": "দয়া করে পাঠ্যটি বাংলায় সারসংক্ষেপ করুন। পাঠ্য: {} সারসংক্ষেপ: ",
  "my": "ကျေးဇူးပြု၍ စာသားကို မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်ပေးပါ။ စာသား: {} အကျဉ်းချုပ်: ",
  "zh": "请用中文（简体）总结文本。文本：{} 总结 ",
  "zt": "請用中文（繁體）總結文本。文本：{} 總結 ",
  "en": "Please summarize the text in English. Text: {} Summary: ",
  "fr": "Veuillez résumer le texte en français. Texte : {} Résumé : ",
  "gu": "કૃપા કરીને પઠનનું સારાંશ ગુજરાતીમાં આપો. પઠન: {} સારાંશ: ",
  "ha": "Da fatan za a taƙaita rubutun da Hausa. Rubutu: {} Taƙaice: ",
  "hi": "कृपया पाठ का सारांश हिंदी में दें। पाठ: {} सारांश: ",
  "ig": "Biko gbakọọ ederede ahụ na Igbo. Ederede: {} Nchịkọta: ",
  "id": "Tolong ringkas teks dalam bahasa Indonesia. Teks: {} Ringkasan: ",
  "ja": "テキストを日本語で要約してください。テキスト: {} 要約: ",
  "rn": "Turinda gusaba ngo usubize inyandiko mu Kirundi. Inyandiko: {} Incamake: ",
  "ko": "텍스트를 한국어로 요약해 주세요. 텍스트: {} 요약: ",
  "ky": "Текстти кыргыз тилинде кыскача баяндагыла. Текст: {} Кыскача баян: ",
  "mr": "कृपया मजकुराचा सारांश मराठीत द्या. मजकूर: {} सारांश: ",
  "ne": "कृपया पाठको सारांश नेपालीमा दिनुहोस्। पाठ: {} सारांश: ",
  "om": "Maqaa galmee dhiheessii afaan Oromoo. Galmee: {} Gabaabaa: ",
  "ps": "مهرباني وکړئ متن په پښتو کې خلاصه کړئ. متن: {} خلاصه: ",
  "fa": "لطفاً متن را به فارسی خلاصه کنید. متن: {} خلاصه: ",
  "pi": "Abeg summarize di text for Pidgin. Text: {} Summary: ",
  "pt": "Por favor, resuma o texto em português. Texto: {} Resumo: ",
  "pa": "ਕਿਰਪਾ ਕਰਕੇ ਪਾਠ ਦਾ ਸਾਰ ਪੰਜਾਬੀ ਵਿੱਚ ਦਿਓ। ਪਾਠ: {} ਸਾਰ: ",
  "ru": "Пожалуйста, кратко изложите текст на русском языке. Текст: {} Резюме: ",
  "gd": "Feuch an toir thu geàrr-chunntas air an teacsa sa Ghàidhlig. Teacsa: {} Geàrr-chunntas: ",
  "sr": "Молимо вас сумирајте текст на српском (ћирилицом). Текст: {} Резиме: ",
  "sl": "Молимо вас сажмите текст на српском (латиницом). Текст: {} Резиме: ",
  "si": "කරුණාකර පෙළ සංක්ෂේප සිංහලෙන් කරන්න. පෙළ: {} සංක්ෂේපය: ",
  "so": "Fadlan nuxurka qoraalka ku soo koob Af-Soomaali. Qoraalka: {} Nuxur: ",
  "es": "Por favor, resuma el texto en español. Texto: {} Resumen: ",
  "sw": "Tafadhali fupisha maandishi kwa Kiswahili. Maandishi: {} Muhtasari: ",
  "ta": "தயவுசெய்து உரையை தமிழில் சுருக்கவும். உரை: {} சுருக்கம்: ",
  "te": "దయచేసి వచనాన్ని తెలుగు లో సారాంశం ఇవ్వండి. వచనం: {} సారాంశం: ",
  "th": "กรุณาสรุปข้อความเป็นภาษาไทย ข้อความ: {} สรุป: ",
  "ti": "እባክዎ ጽሑፉን በትግርኛ ይዘምኑ። ጽሑፍ፡ {} ማጠቃለያ፡ ",
  "tr": "Lütfen metni Türkçe özetleyin. Metin: {} Özet: ",
  "uk": "Будь ласка, підсумуйте текст українською мовою. Текст: {} Резюме: ",
  "ur": "براہ کرم متن کا خلاصہ اردو میں کریں۔ متن: {} خلاصہ: ",
  "uz": "Matnni o'zbek tilida qisqacha yozing. Matn: {} Xulosa: ",
  "vi": "Vui lòng tóm tắt văn bản bằng tiếng Việt. Văn bản: {} Tóm tắt: ",
  "cy": "Crynhoi'r testun yn Gymraeg os gwelwch yn dda. Testun: {} Crynodeb: ",
  "yo": "Jọwọ akopọ ọrọ naa ni Yorùbá. Òrọ̀: {} Akopọ: "
}

is_char_in_lang = {
    "am": lambda char: '\u1200' <= char <= '\u137F',  # Amharic 
    "ar": lambda char: '\u0600' <= char <= '\u06FF',  # Arabic 
    "az": lambda char: '\u0041' <= char <= '\u007A' or '\u00C0' <= char <= '\u00FF',  # Azerbaijani (Latin) 
    "bn": lambda char: '\u0980' <= char <= '\u09FF',  # Bengali 
    "my": lambda char: '\u1000' <= char <= '\u109F',  # Burmese 
    "zh": lambda char: '\u4E00' <= char <= '\u9FFF',  # Chinese Simplified 
    "zt": lambda char: ('\u4E00' <= char <= '\u9FFF') or ('\u3400' <= char <= '\u4DBF') or ('\uF900' <= char <= '\uFAFF'),  # Chinese Traditional 
    "en": lambda char: 'a'<= char <= 'z' or 'A' <= char <= 'Z',  # English 
    "fr": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in"éèêëàâîôûç",  # French 
    "gu": lambda char: '\u0A80' <= char <= '\u0AFF',  # Gujarati 
    "ha": lambda char: 'a' <= char <= 'z'or 'A' <= char <= 'Z',  # Hausa 
    "hi": lambda char: '\u0900' <= char <= '\u097F',  # Hindi 
    "ig": lambda char: 'a' <= char <= 'z'or 'A' <= char <= 'Z',  # Igbo 
    "id": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Indonesian 
    "ja": lambda char: '\u3040' <= char <= '\u30FF' or '\uFF66' <= char <= '\uFF9F' or '\uFF66' <= char <= '\uFF9F' or '\u4E00' <= char <= '\u9FFF',  # Japanese 
    "ki": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Kirundi 
    "ko": lambda char: '\uAC00' <= char <= '\uD7AF',  # Korean 
    "ky": lambda char: '\u0430' <= char <= '\u044F' or '\u0400' <= char <= '\u042F',  # Kyrgyz (Cyrillic) 
    "mr": lambda char: '\u0900' <= char <= '\u097F',  # Marathi
    "ne": lambda char: '\u0900' <= char <= '\u097F',  # Nepali 
    "om": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Oromo 
    "ps": lambda char: '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F',  # Pashto 
    "fa": lambda char: '\u0600' <= char <= '\u06FF',  # Persian 
    "pi": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Pidgin 
    "pt": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in "çáâãéêíóôú",  # Portuguese 
    "pa": lambda char: '\u0A00' <= char <= '\u0A7F',  # Punjabi 
    "ru": lambda char: '\u0410' <= char <= '\u044F',  # Russian 
    "gd": lambda char: 'a' <= char <= 'z' or 'A'<= char <= 'Z' or char in "áàâäéèêëíìîïóòôöúùûü",  # Scottish Gaelic 
    "sc": lambda char: '\u0400' <= char <= '\u045F',  # Serbian Cyrillic 
    "sl": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Serbian Latin 
    "si": lambda char: '\u0D80' <= char <= '\u0DFF',  # Sinhala 
    "so": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Somali 
    "es": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in "áéíóúñ",  # Spanish 
    "sw": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z',  # Swahili 
    "ta": lambda char: '\u0B80' <= char <= '\u0BFF',  # Tamil 
    "te": lambda char: '\u0C00' <= char <= '\u0C7F',  # Telugu 
    "th": lambda char: '\u0E00' <= char <= '\u0E7F',  # Thai 
    "ti": lambda char: '\u1200' <= char <= '\u137F',  # Tigrinya
    "tr": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in "çğıöşü",  # Turkish 
    "uk": lambda char: '\u0400' <= char <= '\u045F',  # Ukrainian 
    "ur": lambda char: '\u0600' <= char <= '\u06FF',  # Urdu 
    "uz": lambda char: 'a' <= char <= 'z' or'A' <= char <= 'Z',  # Uzbek 
    "vi": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in"áàảãạâấầẩẫậêếềểễệíìĩịóòỏõọôốồổỗộúùủũụýỳỷỹỵ",  # Vietnamese 
    "cy": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in "âêîôŵŷ",  # Welsh 
    "yo": lambda char: 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char in "ẹọṣń",  # Yoruba
    "num": lambda char: bool(re.match(r'^\d+(\.\d+)?$', char)),  # Number check
    "sp": lambda char: bool(re.match(r'^[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~。、,，！？：；“”‘’《》【】（）〔〕「」『』\n\r\t\x00\x1b ]+$', char)) # Special characters check
}
