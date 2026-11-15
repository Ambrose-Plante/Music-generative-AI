import csv
import numpy as np

word_list = []

for line in open('words.txt', 'r'):
  word_list.append(line[:-1])

def double_letter_word(word):
  letterlist = [x for x in word]
  for letter in np.unique(letterlist):
    matches = [i for i,x in enumerate(letterlist) if x==letter]
    if len(matches)==2:
      if np.abs(matches[1]-matches[0])==1: ## double letter word
        new_word = word[:matches[0]] + word[matches[1]:]
        new_word = new_word[::-1]
        return new_word

  return False

for word in word_list:
  potential_hit = double_letter_word(word)
  if potential_hit:
    if potential_hit in word_list:
      print(word, '/', potential_hit)

'accreditation' in word_list
