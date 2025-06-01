from collections import Counter
import re

def tokenizer(text,top_k='all_words'):  
   # remove punctuation and non alphabetic characters
   remove_punctuation = re.sub(r'[^\w\s]', '', text)
   lower_case_words = remove_punctuation.lower()
   words = lower_case_words.split()

   print(f"Number of words before filtering: {len(words)}")

   # get word counts
   word_counts = Counter(words)
   if top_k=='all_words':
      top_words = dict(word_counts)
   else:
      top_words = dict(word_counts.most_common(top_k))
   word_to_id = {word: i for i, word in enumerate(top_words.keys())}
   id_to_word = {i: word for i, word in enumerate(top_words.keys())}
   print("total number of unique words:",len(word_counts))

   total_count = sum(count for word, count in top_words.items())

   if top_k=='all_words':
      print(f"Total count of all words: {total_count}")
   else:   
      print(f"Total count of top {top_k} words: {total_count}")
   # Optional: Show what percentage of all words this represents
   total_words = sum(word_counts.values())
   percentage = (total_count / total_words) * 100
   print(f"This represents {percentage:.2f}% of all words in the corpus")
   # filter corpus to only include words in the tok k words
   corpus = [word for word in words if word in top_words]
   print("corpus length:", len(corpus))
   print("Number of unique words:", len(word_to_id))
   return word_to_id, id_to_word, corpus

if __name__ == "__main__":
   # Example usage
   #text = "This is a sample text with some words. Some words are repeated, and some are not."
   top_k = 'all_words'  # or any integer value
   word_to_id, id_to_word, corpus = tokenizer(text, top_k=top_k)
   print("top_k:", top_k)
   print("Number of unique words:", len(word_to_id))
   print("Filtered corpus:", corpus)
