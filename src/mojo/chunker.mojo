# =============================================================================
# COMPLEXITY NOTES
# =============================================================================
# split_words():      O(L) где L — длина текста в байтах
# chunk_by_words():   O(W) где W — число слов
# chunk_text():       O(L + W)
#
# Для файлов > 10MB рассмотреть потоковый chunking (фаза 2)
# =============================================================================

struct Chunk(Copyable, Movable, Writable):
    var text: String
    var start: Int
    var end: Int
    
    def __init__(out self, text: String, start: Int, end: Int):
        self.text = text
        self.start = start
        self.end = end
    
    def __init__(out self, *, copy: Self):
        self.text = copy.text
        self.start = copy.start
        self.end = copy.end
    
    def write_to(self, mut writer: Some[Writer]):
        writer.write("Chunk(start=", self.start, ", end=", self.end, ")")


def split_words(text: String) -> List[String]:
    var words = List[String]()
    var current = String("")
    
    for i in range(len(text)):
        var ch = text[byte=i]
        var ch_str = String(ch)
        
        if ch_str == " " or ch_str == "\n" or ch_str == "\t":
            if len(current) > 0:
                words.append(current^)
                current = String("")
        else:
            current += ch_str
    
    if len(current) > 0:
        words.append(current^)
    
    return words^


def chunk_by_words(
    text: String,
    chunk_size: Int,
    overlap: Int,
) -> List[Chunk]:
    var words = split_words(text)
    var chunks = List[Chunk]()
    
    if len(words) == 0:
        return chunks^
    
    var step = chunk_size - overlap
    if step <= 0:
        step = 1
    
    var pos = 0
    while pos < len(words):
        var end = pos + chunk_size
        if end > len(words):
            end = len(words)
        
        var chunk_text = String("")
        for i in range(pos, end):
            if i > pos:
                chunk_text += " "
            chunk_text += words[i]
        
        if len(chunk_text) > 0:
            chunks.append(Chunk(text=chunk_text, start=pos, end=end))
        
        pos += step
        
        if pos >= len(words):
            break
    
    return chunks^


def chunk_text(
    text: String,
    chunk_size: Int,
    overlap: Int,
) -> List[Chunk]:
    var result = chunk_by_words(text, chunk_size, overlap)
    return result^
