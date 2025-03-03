# Curriculum Learning

By Luc(i?)a

## Models

- 5 Random Models
- Curriculum Sentence Length
- Curriculum Gulpease
- Curriculum Readit

### 3 model initialization seeds

 - Seed 42 🌱  
 - Seed 995 🌷  
 - Seed 755 🥑

 - Seed 42 + Eye-tracking 🥕

## Things to do

Model performances 🏋️
 
|                | Probing | PPL Wiki | PPL Treebank | Sentiment | Complexity | POS Tagging |
|----------------|---------|----------|--------------|-----------|------------|-------------|
| Orig           | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Orig_I         | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Rand42         | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Rand42_I       | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandPI         | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandPI_I       | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandR2         | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandR2_I       | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandG          | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| RandG_I        | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Gulpease       | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Gulpease_I     | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Sentence_len   | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Sentence_len_I | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Readit         | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |
| Readit_I       | 🌱  🌷  🥑  🥕 | 🌱 🌷  🥑  🥕 | 🌱  🌷  🥑  🥕 | 🌱  🌷  🥑 | 🌱  🌷  🥑 | 🌱  🌷  🥑 |



# Representations' Space 🚀

|                | IsoScore     | Lin_ID       | NonLin_ID      | cos_sim       | varex         | Partition (?)   |
|----------------|--------------|--------------|----------------|---------------|---------------|-----------------|
| Orig           | 🌱  🌷 🥑    | 🌱   🥑     | 🌱   🌷  🥑    | 🌱   🌷 🥑   | 🌱   🌷 🥑  | 🌷               |
| Orig_I         | 🌱 🌷        | 🌱 🌷         | 🌱             | 🌱   🌷      | 🌱 🌷       | 🌱                |
| Rand42         | 🌱 🌷         | 🌱🌷         | 🌱 🌷          | 🌱 🌷         | 🌱 🌷        |🌱              |
| Rand42_I       | 🌱 🌷         🌱 🌷          | 🌱  🌷         | 🌱 🌷         |🌱  🌷        |🌱              |
| RandPI         | 🌱           | 🌱🌷          | 🌱 🌷           | 🌱 🌷        | 🌱 🌷         | 🌱              |
| RandPI_I       | 🌱           | 🌱           | 🌱             | 🌱            | 🌱            | 🌱              |
| RandR2         | 🌱🌷         | 🌱🌷         | 🌱🌷           | 🌱🌷          | 🌱🌷          | 🌱              |
| RandR2_I       | 🌱🌷         | 🌱 🌷         | 🌱 🌷           | 🌱 🌷         | 🌱🌷        | 🌱            |
| RandG          | 🌱           | 🌱           | 🌱             | 🌱            | 🌱            | 🌱              |
| RandG_I        | 🌱           | 🌱           | 🌱             | 🌱            | 🌱            | 🌱              |
| Gulpease       | 🌱  🌷 🥑    | 🌱  🌷 🥑     | 🌱   🌷 🥑    | 🌱   🌷 🥑    | 🌱  🌷 🥑     |                |
| Gulpease_I     | 🌱   🌷 🥑   | 🌱  🌷  🥑    | 🌱   🌷 🥑     | 🌱  🌷  🥑    | 🌱   🌷🥑    |                 |
| Sentence_len   | 🌱  🌷 🥑    |   🌷 🥑    | 🌱 🌷  🥑      | 🌱 🌷   🥑    | 🌱  🌷 🥑     | 🌱             |
| Sentence_len_I | 🌱  🌷       | 🌱         | 🌱 🌷          | 🌱  🌷        | 🌱 🌷         | 🌱            |
| Readit         | 🌱  🌷 🥑    | 🌱  🌷 🥑     | 🌱  🌷 🥑      | 🌱  🌷   🥑    | 🌱  🌷🥑    | 🌱             |
| Readit_I       | 🌱   🌷       | 🌱  🌷        | 🌱  🌷          | 🌱  🌷         | 🌱   🌷     | 🌱            |


