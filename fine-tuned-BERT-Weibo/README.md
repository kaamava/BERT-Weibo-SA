# Fine-tune BERT for sentiment analysis on Weibo
Performing sentiment analysis on Chinese Weibo comments using BERT.

1. Utilize the pre-trained model "chinese-bert-wwm-ext." Create a directory named "chinese-bert-wwm-ext" in the project directory to store the pre-trained model downloaded from Hugging Face.
2. In the project directory, create a directory named "model." Execute the "main.py" script for fine-tuning, and the trained model will be saved in the "model" directory.
3. Modify the model loading path in the "config.py" file. Run the "inference.py" script for inference. The output of "inference.py" is a numerical value between [-0.5, 0.5], where a higher value indicates a higher likelihood of positivity and a lower value suggests a higher likelihood of negativity.
4. If you wish to directly output the classification result, simply remove the argmax judgment statement from the "predict" function in "inference.py." The result of argmax being 1 indicates a likelihood of positivity, and 0 indicates a likelihood of negativity. Using NLP terminology, this process involves sentiment analysis of Chinese microblog comments based on BERT.
