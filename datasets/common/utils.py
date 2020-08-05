"""
去掉人民日报语料里面的中括号以及中括号里面词的词性
"""
def rm_replace_symbol(symbols: list, news_str):
    start = news_str.find(symbols[0])
    end = news_str.find(symbols[1])
    word = "".join([x.split("/")[0] for x in news_str[start + 1:end].split(" ")])
    tag = news_str[end + 1:end + 3]
    newsparagraph = news_str.replace(news_str[start:end + 3], word + "/" + tag)
    while newsparagraph.find("[") >= 0 and newsparagraph.find("]") > 0:
        newsparagraph = rm_replace_symbol(["[", "]"], newsparagraph)
    return newsparagraph


"""
words = ["今天","心情"，“好]~
"""
def segment_to_bmes(words):
    new_words = []
    new_tags = []
    for word in words:
        if len(word) == 1:
            new_words.append(word)
            new_tags.append("S")
        elif len(word) > 1:
            new_words.append(word[0])
            new_tags.append("B")
            for w in word[1:len(word) - 1]:
                new_words.append(w)
                new_tags.append("M")
            new_words.append(word[len(word) - 1])
            new_tags.append("E")
        else:
            continue
    return new_words, new_tags

def format_sentence(sentence: str):
    return " ".join(sentence.strip().replace(" ", ""))