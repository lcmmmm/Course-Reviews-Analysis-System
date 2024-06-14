no_con = [] # 沒有總結的 id
X_train = []
for i in range(2012):
    with open(f"./data/post_{i+1}.txt", 'r', encoding='utf-8') as file:
        text = file.read()
        match = re.search(r'Ψ 總結(.*?)(?=--|$)', text, re.DOTALL)
        if match:
            conclusion_text = match.group(1).strip()
            X_train.append(conclusion_text)
        else:
            no_con.append(i+1)
print(len(no_con))
print(no_con)

import os
import re

def calculate_total_stars(review_text):
    # Define a regex pattern for stars
    pattern = re.compile(r'[★]')
    
    # Count the number of stars in the review text
    star_count = len(pattern.findall(review_text))
    
    return star_count

def calculate_adjusted_average_stars(paragraph_text, stars_in_same_line):
    # Define a regex pattern for stars
    star_pattern = re.compile(r'[★]+')
    
    # Find all matches in the paragraph text
    matches = star_pattern.findall(paragraph_text)
    initial = [min(5, len(match)) for match in matches]
    adjusted = max(0, sum(initial))
    # print(adjusted)
    num_ratings = 0
    
    
    # If stars are present in the same line and not exactly 5, calculate the average without deducting 5
    if stars_in_same_line and (adjusted >= 1 and adjusted <= 4):
        total_stars = adjusted
        # print(total_stars)
        num_ratings = 1
    
    else:
        if stars_in_same_line and adjusted == 5: # ★★★★★
            # Check if the paragraph contains only stars (e.g., ★★★★★)
            if re.fullmatch(r'[\s★☆]+', paragraph_text):
                # print(paragraph_text)
                total_stars = 5
            else:
                # Cap original star ratings at 5
                original_stars = [min(5, len(match)) for match in matches]

                # Deduct five stars from the total count
                total_stars = max(0, sum(original_stars) - 5)
                # num_ratings = max(1, len(matches) - 1)
                
        else: # no stars
            # Check if the paragraph contains only stars (e.g., ★★★★★)
            if re.fullmatch(r'[\s★☆]+', paragraph_text):
                # print(paragraph_text)
                total_stars = 5
            else:
                # Cap original star ratings at 5
                original_stars = [min(5, len(match)) for match in matches]

                # Deduct five stars from the total count
                total_stars = max(0, sum(original_stars))
                
        # If the total stars are still less than 5, check for Chinese numerals and Arabic numerals
        if total_stars < 5:
            # Check for Chinese numerals in the paragraph and calculate their average
            chinese_numerals = re.findall(r'[一二三四五]', paragraph_text)
            total_stars += sum([1 if numeral == '一' else 2 if numeral == '二' else 3 if numeral == '三'
                                else 4 if numeral == '四' else 5 for numeral in chinese_numerals])
            num_ratings += len(chinese_numerals)
            # print(chinese_numerals)

            # Check for Arabic numerals in the paragraph and calculate their average
            arabic_numerals = re.findall(r'[1-5]\.?\d*', paragraph_text)
            total_stars += sum([min(5, float(numeral)) for numeral in arabic_numerals])
            num_ratings += len(arabic_numerals)

            # Check 滿天星 & 無限
            inf_numerals = re.findall(r'[滿天星]', paragraph_text)
            total_stars += sum([5 for numeral in inf_numerals])
            num_ratings += len(inf_numerals)

            infi_numerals = re.findall(r'[無限]', paragraph_text)
            total_stars += sum([5 for numeral in infi_numerals])
            num_ratings += len(infi_numerals)
    
    if num_ratings == 0:
        num_ratings = max(1, len(matches))
    
    adjusted_average_stars = total_stars / num_ratings
    return adjusted_average_stars


label = []
text = []

def process_reviews_in_directory(directory_path, no_con):
    # Counter for files with adjusted average stars equal to 0
    zero_stars_count = 0
    
    total_result = 0
    count = 0
    no_comment = 0
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            count += 1
            # Extract document ID from the filename
            doc_id = int(filename.split('_')[1].split('.')[0])
            
            # Skip if the document ID is in the no_con list
            if doc_id in no_con:
                # print(doc_id)
                continue
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read()
           
                # Try to find the paragraph between "Ω 私心推薦指數(以五分計)" and "η 上課用書(影印講義或是指定教科書)"
                paragraph_pattern = re.compile(r'Ω 私心推薦指數\(以五分計\)(.*?)η 上課用書\(影印講義或是指定教科書\)', re.DOTALL)
                paragraph_match = paragraph_pattern.search(review_text)
                
                # If not found, try the alternative paragraph marker
                if not paragraph_match:
                    paragraph_pattern_alternative = re.compile(r'Ω 私心推薦指數（以五分計）(.*?)η 上課用書（影印講義或是指定教科書）', re.DOTALL)
                    paragraph_match = paragraph_pattern_alternative.search(review_text)
                    no_comment += 1

                if paragraph_match:
                    # Extract the content of the paragraph
                    paragraph_text = paragraph_match.group(1)
                    
                    # Check if stars are present in the same line as "Ω 私心推薦指數(以五分計)" or "Ω 私心推薦指數（以五分計）"
                    # stars_in_same_line = re.search(r'Ω 私心推薦指數\(以五分計\).*?[★]+', paragraph_text) or re.search(r'Ω 私心推薦指數（以五分計）.*?[★]+', paragraph_text)
                    same_line = re.compile(r'Ω 私心推薦指數\(以五分計\)\s*([★☆]+)', re.IGNORECASE)
                    match = same_line.search(review_text)
                    if match:
                        stars_in_same_line = True
                    else:
                        stars_in_same_line = False
                    # print(stars_in_same_line)
                    # Calculate adjusted average stars for the paragraph
                    adjusted_average_stars = round(calculate_adjusted_average_stars(paragraph_text, stars_in_same_line))
                    if adjusted_average_stars == 0:
                        adjusted_average_stars = 1
                        
                    label.append(adjusted_average_stars)
                    match = re.search(r'Ψ 總結(.*?)(?=--|$)', review_text, re.DOTALL)
                    conclusion_text = match.group(1).strip()
                    text.append(conclusion_text)
                    
                    # Print or store the results as needed
                    # print(f"File: {filename}, Adjusted Average Stars: {adjusted_average_stars:.2f}")
                    total_result += 1
                    # if adjusted_average_stars > 5:
                        # print(f"File: {filename}, ahhhhhhhhhhhhhhhhhhhhhhhhhh: {adjusted_average_stars:.2f}")
                    
                    # Check if adjusted average stars are 0
                    if adjusted_average_stars == 0:
                        zero_stars_count += 1
    
    # Print the number of files with adjusted average stars equal to 0
    # print(f"Number of files with adjusted average stars equal to 0: {zero_stars_count}")
    # print(total_result)
    # print(count)
    # print(no_comment)
# Specify the directory path
data_directory = './data'

# Process all reviews in the specified directory
process_reviews_in_directory(data_directory, no_con)
# print(len(no_con))

print(len(label))
print(len(text))

from datasets import Dataset, DatasetDict

train_data = {'label': label[:1560], 'text': text[:1560]}
test_data = {'label': label[1560:], 'text': text[1560:]}

# Create Dataset objects
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Create DatasetDict
dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Print information about the constructed dataset dictionary
print(dataset_dict)

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch

import re

# 讀取檔案內容
file_path = './data/post_923.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    review_text = file.read()

# 抓取 "Ω 私心推薦指數(以五分計)" 該行內容
pattern = re.compile(r'Ω 私心推薦指數\(以五分計\)\s*([★☆]+)', re.IGNORECASE)
match = pattern.search(review_text)

if match:
    # 抓取到的 "Ω 私心推薦指數(以五分計)" 該行內容
    line_content = match.group(0).strip()
    print(line_content)
else:
    print("未找到相應的內容")


import os
import re

def is_paragraph_empty(review_text):
    # Try to find the paragraph between "Ω 私心推薦指數(以五分計)" and "η 上課用書(影印講義或是指定教科書)"
    paragraph_pattern = re.compile(r'Ω 私心推薦指數\(以五分計\)(.*?)η 上課用書\(影印講義或是指定教科書\)', re.DOTALL)
    paragraph_match = paragraph_pattern.search(review_text)
    
    # If not found, try the alternative paragraph marker
    if not paragraph_match:
        paragraph_pattern_alternative = re.compile(r'Ω 私心推薦指數（以五分計）(.*?)η 上課用書（影印講義或是指定教科書）', re.DOTALL)
        paragraph_match = paragraph_pattern_alternative.search(review_text)

    if paragraph_match:
        # Extract the content of the paragraph
        paragraph_text = paragraph_match.group(1)
        
        # Check if the paragraph is empty or contains only whitespace
        is_empty = not paragraph_text or not paragraph_text.strip()
        
        if not is_empty:
            # Print the content of the non-empty paragraph
            print(f"Non-empty paragraph content:\n{paragraph_text}")
        
        return is_empty
    else:
        # If the pattern is not found, consider the paragraph as empty
        return True

# Specify the file path
file_path = './data/post_995.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    review_text = file.read()

# Check if the paragraph is empty
is_empty = is_paragraph_empty(review_text)

# Print the result
print(f"The paragraph is empty: {is_empty}")

import nltk
nltk.download('wordnet')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    # 移除特殊符號和標點符號
    text = re.sub(r'[^\w\s]', '', text)
    
    # 轉換為小寫
    text = text.lower()
    
    # 切詞
    tokens = word_tokenize(text)
    
    # 移除停用詞
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 詞形還原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# 測試前處理
example_text = "還是推薦這堂課，玉琦和助教都對學生很友善，並且能感覺到老師很積極想要提升教學品質，我相信會越來越好啦。"
preprocessed_text = preprocess(example_text)
print(preprocessed_text)
