# Curriculum Learning

By Luc(i?)a

## Models

- 5 Random Models
- Curriculum Sentence Length
- Curriculum Gulpease
- Curriculum Readit

## Things to do

Model performances 🏋️
 
|                | Probing | PPL Wiki | PPL Treebank | Sentiment | Complexity | POS Tagging |
|----------------|---------|----------|--------------|-----------|------------|-------------|
| Orig           | X       | X        | X            | X         | X          | X           |
| Orig_I         | X       | X        | X            | X         | X          | X           |
| Rand42         | X       | X        | X            | X         | X          | X           |
| Rand42_I       | X       | X        | X            | X         | X          | X           |
| RandPI         | X       | X        | X            | X         | X          | X           |
| RandPI_I       | X       | X        | X            | X         | X          | X           |
| RandR2         | X       | X        | X            | X         | X          | X           |
| RandR2_I       | X       | X        | X            | X         | X          | X           |
| RandG          |         |          |              |           |            |             |
| RandG_I        |         |          |              |           |            |             |
| Gulpease       | X       | X        | X            | X         | X          | X           |
| Gulpease_I     | X       | X        | X            | X         | X          | X           |
| Sentence_len   | X       | X        | X            | X         | X          | X           |
| Sentence_len_I | X       | X        | X            | X         | X          | X           |
| Readit         | X       | X        | X            | X         | X          | X           |
| Readit_I       | X       | X        | X            | X         | X          | X           |


Representations'space 🚀
SEED42 🐭

|                | IsoScore | Lin_ID  | NonLin_ID    | cos_sim   | varex       | Partition (?) | 
|----------------|---------|----------|--------------|-----------|------------|-------------| 
| Orig           | X       |   X      | X            | X         | X          |             |
| Orig_I         | X       |   X      | X            | X         | X          |  X          |
| Rand42         | X       |   X      |  X           | X         | X          |  X          |
| Rand42_I       | X       |   X      |  X           | X         | X          | X           |
| RandPI         | X       |   x      |  X           | X         | X          | X           |
| RandPI_I       | X       |   X      |  X           | X         | X          | X           |
| RandR2         |X        |   X      |  x           | X         | X          |X            |
| RandR2_I       |X        |   X      |  X           | X         | X          |X            |
| RandG          |X        |   X      |  X           | X         | X          |X            | -----> Done (almost)!
| RandG_I        |X        |   X      |  X           |  X        |    X       |X            |
| Gulpease       |X        |   X      |   X          |  X        |    X       |             |
| Gulpease_I     |X        |   X      |   X          |  X        |    X       |             |
| Sentence_len   |X        |   X      |   X          |  X        |    X       | X           |
| Sentence_len_I |X        |   X      |   x          |  X        |    x       | X           |
| Readit         |X        |   X      |   X          |  X        |    X       | X           |
| Readit_I       |X        |   X      |   X          |  X        |    X       | X           |

SEED995 🕵️‍♀️

Here’s a cleaner and better-organized version of the table:

| **Metric**       | **IsoScore** | **Lin_ID** | **NonLin_ID** | **cos_sim** | **varex** | **Partition (?)** |  
|-------------------|--------------|------------|---------------|-------------|-----------|--------------------|  
| **Orig**          | x            | x          | x             | x           | x         | x                  |  
| **Orig_I**        |              |            |               |             |           |                    |  
| **Rand42**        |              |            |               |             |           |                    |  
| **Rand42_I**      |              |            |               |             |           |                    |  
| **RandPI**        |              |            |               |             |           |                    |  
| **RandPI_I**      |              |            |               |             |           |                    |  
| **RandR2**        |              |            |               |             |           |                    |  
| **RandR2_I**      |              |            |               |             |           |                    |  
| **RandG**         |              |            |               |             |           |                    |  
| **RandG_I**       |              |            |               |             |           |                    |  
| **Gulpease**      | x            |   x        |   x           |  x          | x         |                    |  
| **Gulpease_I**    |              |            |               |             |           |                    |  
| **Sentence_len**  |  x           |    x       |             x |   x         | x         |                    |  
| **Sentence_len_I**|              |            |               |             |           |                    |  
| **Readit**        |  x           |   x        |    x          |    x        |     x     |                    |  
| **Readit_I**      |              |            |               |             |           |                    |


