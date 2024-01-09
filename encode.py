from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Encode String Function
# Example 1: Semiconductor device and a method of fabricating the same
# Example 2: A power MOSFET comprises: a semiconductor substrate  21  of a first conduction type; a drain layer  22  of the first conduction 
#   type and formed on a surface layer of the substrate; a gate insulating...
# Encoding Algoritm:
# 1. Join List into single string
# 2. Take out common words ("stopwords")
# 3. Tokenize single string (string to list)
# 4.  Create frequency distribution, excluding non-aplhabetic words and words in filter and get the top 100 words
# 5.  Iterate through strings: 
#       Tokenize string (string to list) and filter
#       Freqency Encoding
#       Add to Encoded Data output
def encodeStrings(data):
    encodedData = []
    data = [str(item) for item in data]
    data_all = ' '.join(data)
    filter = set(stopwords.words('english'))
    data_tokens = word_tokenize(data_all) 
    fdist = FreqDist(word.lower() for word in data_tokens if word.isalpha() and word.lower() not in filter) 
    common_words_100 = fdist.most_common(100)
    top_words = [word for word, freq in common_words_100] 
    for str_data in data:
        str_tokens = word_tokenize(str_data)
        str_filtered_tokens = [word.lower() for word in str_tokens if word.isalpha() and word.lower() not in filter]
        str_word_freq = Counter(str_filtered_tokens)
        row_frequencies = [str_word_freq.get(word, 0) for word in top_words]
        encodedData.append(row_frequencies)
    return encodedData


# Encode UCID Function
# Example: US-6939776-B2
# 1. Make Map, key: String ID, value: unique ID
# 2. Iterate thorugh data:
#       Split data based on '-'
#       Get rid of letters in number, if any
#       Append number and integer tag to data
def encodeUCIDs(data):
    encodedData = []
    tag_map = {'B2': 1, 'B1': 2, 'A1': 3, 'A': 4, 'A2': 5, 'E': 6, 'A9': 7, 'P1': 8, 'P3': 9, 'H': 10, 'P2': 11, 'P': 12, 'S': 13, 'I4': 14}
    for row in data:
        split_data = row.split('-')
        number = int(''.join(filter(str.isnumeric, split_data[1])))
        tag = split_data[2]
        encodedData.append([number, tag_map.get(tag, 0)])
    return np.array(encodedData)

# Encode Codes Function
# Example: H01L29/41766
# 1. Iterate thorugh data:
#       Split data based on '/'
#       Get rid of letters in number, if any
#       Append number and integer tag to data
def encodeCodes(data):
    encodedData = []
    for row in data:
        split_data = row.split('/')
        tag = int(''.join(filter(str.isnumeric, split_data[0])))
        number = int(split_data[1])      
        encodedData.append([tag, number])
    return np.array(encodedData)

# Encode CPCs Function
# Example: H01L
# 1. Iterate thorugh data:
#       if code is not in map, add it; increment counter
#       append the mapped code to the encoded data
def encodeCPCs(data):
    encodedMap = {}
    encodedData = []
    numeric_value = 1
    for code in data:
        mapCode = []
        if code not in encodedMap:
            encodedMap[code] = numeric_value
            numeric_value += 1
        mapCode = encodedMap[code]
        encodedData.append(mapCode)
    return np.array(encodedData)