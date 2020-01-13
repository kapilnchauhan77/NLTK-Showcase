from nltk.tokenize import sent_tokenize, word_tokenize

sentence = "Hi this is me Mr. Kapil.N.Chauhan showcasing nltk,\
 word_tokenizer and sent_tokenize to myself.\
 I am fairly new at this game but I will improve.\
 Also, I cannot type rigth now,\
 I am feeling very cold, my hands cannot move easily!!!"

print("Sentence:")
print(sentence)
print("\nSentences Tokenized:")
print(sent_tokenize(sentence))
print("\nWords Tokenized:")
print(word_tokenize(sentence))
